"""
Custom controller for comma.ai Controls Challenge
Implements CMA-ES optimized feedback control with multi-step weighted lookahead
"""

import numpy as np
from collections import deque
import json
import os
from . import BaseController


class Controller(BaseController):
    """
    CMA-ES Optimized Feedback Controller with Multi-Step Lookahead
    
    This controller uses weighted averaging over multiple future steps
    for smoother acceleration control and better anticipation.
    """
    
    # Default parameters with multi-step lookahead (5 steps: indices 2-6)
    DEFAULT_PARAMS = {
        "kp": 0.4112591450588017,
        "ki": 0.13928150947159526,
        "kd": 0.1323549340551643,
        "k_roll": 0.5103031070180226,
        "k_ff_immediate": 0.032229250569400375,
        "k_ff_lookahead": 0.38537725904093284,
        "lookahead_start_idx": 2.0,
        "lookahead_end_idx": 8.0,
        "lookahead_decay": 0.7527223415032701,
        "error_filter_alpha": 0.2716908688847302,
        "action_smoothing": 0.21346426416784026,
        "velocity_scaling": 0.003967642884832395,
        # New velocity-dependent parameters
        "velocity_kp_scale": 0.0,
        "velocity_ki_scale": 0.0,
        "velocity_kd_scale": 0.0,
        "velocity_ff_scale": 0.0,
        "future_velocity_weight": 0.0,
        "velocity_lookahead_scale": 0.0,
    }
    
    def __init__(self, params=None):
        """Initialize controller with optional parameter override.
        
        Args:
            params: Optional dict of parameters to use. If None, will try to load
                   from optimized_params_multistep.json or fall back to DEFAULT_PARAMS.
        """
        if params is not None:
            # Use provided parameters directly (for CMA-ES optimization)
            self.params = params.copy()
        else:
            # Try to load optimized parameters from file
            params_file = os.path.join(os.path.dirname(__file__), '..', 'optimized_params_velocity.json')
            # params_file = os.path.join(os.path.dirname(__file__), '..', 'optimized_params_multistep.json')
            # params_file = os.path.join(os.path.dirname(__file__), '..', 'optimized_params_multistep_final.json')
            
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    loaded = json.load(f)
                    # Remove metadata fields
                    self.params = {k: v for k, v in loaded.items() 
                                 if not k.startswith('_')}
            else:
                self.params = self.DEFAULT_PARAMS.copy()
        
        # Ensure all new velocity-dependent parameters exist (backward compatibility)
        for key in ['velocity_kp_scale', 'velocity_ki_scale', 'velocity_kd_scale', 
                   'velocity_ff_scale', 'future_velocity_weight', 'velocity_lookahead_scale']:
            if key not in self.params:
                self.params[key] = 0.0
            
        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_error = 0.0
        self.prev_action = 0.0
        self.error_history = deque(maxlen=20)
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Compute steering action using multi-step lookahead.
        
        Args:
            target_lataccel: The target lateral acceleration.
            current_lataccel: The current lateral acceleration.
            state: The current state of the vehicle.
            future_plan: The future plan with lataccel array.
            
        Returns:
            The control signal (steer action).
        """
        p = self.params
        
        # Extract future plan as arrays
        future_array = np.array(future_plan.lataccel) if hasattr(future_plan, 'lataccel') else []
        future_velocity = np.array(future_plan.v_ego) if hasattr(future_plan, 'v_ego') else []
        
        # Get current velocity with validation
        v_ego = state.v_ego if hasattr(state, 'v_ego') else 20.0
        if not np.isfinite(v_ego) or v_ego < 0:
            v_ego = 20.0  # Default to reasonable value if invalid
        
        v_ref = 20.0  # Reference velocity for normalization
        
        # Compute multi-step weighted lookahead target
        lookahead_start = int(p['lookahead_start_idx'])
        lookahead_end = int(p['lookahead_end_idx'])
        
        if len(future_array) > lookahead_start:
            # Weighted average of future targets with exponential decay
            # Optionally weight by future velocity if available
            weights = []
            targets = []
            for i in range(lookahead_start, min(lookahead_end + 1, len(future_array))):
                weight = p['lookahead_decay'] ** (i - lookahead_start)
                
                # Apply velocity-dependent weighting if future velocity is available
                if len(future_velocity) > i and p['velocity_lookahead_scale'] != 0.0:
                    v_future = future_velocity[i]
                    # Check for valid velocity value (not NaN or inf)
                    if np.isfinite(v_future) and v_ref > 0:
                        # Weight higher when velocity is closer to reference (more predictable)
                        v_factor = 1.0 - p['velocity_lookahead_scale'] * abs(v_future - v_ref) / v_ref
                        weight *= max(0.1, v_factor)  # Prevent negative weights
                
                weights.append(weight)
                targets.append(future_array[i])
            
            if weights:
                lookahead_target = np.average(targets, weights=weights)
            else:
                lookahead_target = target_lataccel
        else:
            lookahead_target = target_lataccel
        
        # Compute velocity-dependent scaling factors
        # Ensure v_ref > 0 to avoid division by zero
        if v_ref <= 0:
            v_ref = 20.0
        v_normalized = (v_ego - v_ref) / v_ref if v_ref > 0 else 0.0  # Normalized velocity deviation
        
        # Overall velocity scaling (backward compatible)
        velocity_factor = 1.0 + p['velocity_scaling'] * (v_ego - v_ref)
        
        # Separate velocity-dependent scaling for each PID term
        kp_velocity_factor = 1.0 + p['velocity_kp_scale'] * v_normalized
        ki_velocity_factor = 1.0 + p['velocity_ki_scale'] * v_normalized
        kd_velocity_factor = 1.0 + p['velocity_kd_scale'] * v_normalized
        
        # Future velocity-aware feedforward scaling
        if len(future_velocity) > 0 and p['future_velocity_weight'] != 0.0:
            # Average future velocity in lookahead window
            v_slice_start = min(lookahead_start, len(future_velocity))
            v_slice_end = min(lookahead_end + 1, len(future_velocity))
            if v_slice_end > v_slice_start:
                v_slice = future_velocity[v_slice_start:v_slice_end]
                # Filter out NaN and inf values
                v_slice_valid = v_slice[np.isfinite(v_slice)]
                if len(v_slice_valid) > 0:
                    v_future_avg = np.mean(v_slice_valid)
                    if v_ref > 0:
                        v_future_factor = 1.0 + p['future_velocity_weight'] * (v_future_avg - v_ref) / v_ref
                    else:
                        v_future_factor = 1.0
                else:
                    # No valid values, use current velocity instead
                    v_future_factor = 1.0 + p['future_velocity_weight'] * (v_ego - v_ref) / v_ref if v_ref > 0 else 1.0
            else:
                # Empty slice, use current velocity instead
                v_future_factor = 1.0 + p['future_velocity_weight'] * (v_ego - v_ref) / v_ref if v_ref > 0 else 1.0
        else:
            v_future_factor = 1.0
        
        # Compute errors
        error_immediate = target_lataccel - current_lataccel
        
        # Filter the error to reduce noise sensitivity
        self.filtered_error = (
            p['error_filter_alpha'] * error_immediate +
            (1 - p['error_filter_alpha']) * self.filtered_error
        )
        self.error_history.append(error_immediate)
        
        # PID terms on filtered error with velocity-dependent scaling
        p_term = p['kp'] * self.filtered_error * kp_velocity_factor
        
        # Integral with anti-windup and velocity-dependent scaling
        self.integral += self.filtered_error
        integral_limit = 2.0 / max(p['ki'], 0.001)
        self.integral = np.clip(self.integral, -integral_limit, integral_limit)
        i_term = p['ki'] * self.integral * ki_velocity_factor
        
        # Derivative using error history for robustness with velocity-dependent scaling
        if len(self.error_history) >= 2:
            derivative = self.error_history[-1] - self.error_history[-2]
        else:
            derivative = 0.0
        d_term = p['kd'] * derivative * kd_velocity_factor
        
        # Feedforward terms with velocity-dependent scaling
        roll_lataccel = state.roll_lataccel if hasattr(state, 'roll_lataccel') else 0.0
        roll_ff = -p['k_roll'] * roll_lataccel
        target_ff_immediate = p['k_ff_immediate'] * target_lataccel * (1.0 + p['velocity_ff_scale'] * v_normalized)
        target_ff_lookahead = p['k_ff_lookahead'] * lookahead_target * v_future_factor
        
        # Combine all terms
        raw_action = (p_term + i_term + d_term + 
                      roll_ff + target_ff_immediate + target_ff_lookahead)
        
        # Action smoothing to reduce jerk
        smoothed_action = (
            p['action_smoothing'] * self.prev_action +
            (1 - p['action_smoothing']) * raw_action
        )
        
        # Clip to valid range
        action = np.clip(smoothed_action, -2.0, 2.0)
        
        # Update state
        self.prev_error = error_immediate
        self.prev_action = action
        
        return action