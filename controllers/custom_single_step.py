"""
Custom controller for comma.ai Controls Challenge
Implements CMA-ES optimized feedback control strategy
"""

import numpy as np
from collections import deque
import json
import os
from . import BaseController


class Controller(BaseController):
    """
    CMA-ES Optimized Feedback Controller
    
    This controller uses a feature-rich feedback structure optimized
    via CMA-ES for the comma.ai controls challenge.
    """
    
    # Default parameters 
    DEFAULT_PARAMS = {
      "kp": 0.5469135358044432,
      "ki": 0.11529377410832296,
      "kd": 0.13687237415493828,
      "k_roll": 0.6450220132120998,
      "k_ff_immediate": 0.10218481706232792,
      "k_ff_lookahead": 0.3179298967048029,
      "lookahead_idx": 3.166047677078586,
      "error_filter_alpha": 0.1866964666300676,
      "action_smoothing": 0.31184410942578455,
      "velocity_scaling": 0.04441798583663417,
    }
    
    def __init__(self, params=None):
        """Initialize controller with optional parameter override.
        
        Args:
            params: Optional dict of parameters to use. If None, will try to load
                   from optimized_params.json or fall back to DEFAULT_PARAMS.
        """
        if params is not None:
            # Use provided parameters directly (for CMA-ES optimization)
            self.params = params.copy()
        else:
            # Try to load optimized parameters from file
            params_file = os.path.join(os.path.dirname(__file__), '..', 'optimized_params.json')
            
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    self.params = json.load(f)
                    # Remove metadata fields
                    self.params = {k: v for k, v in self.params.items() if not k.startswith('_')}
            else:
                self.params = self.DEFAULT_PARAMS.copy()
            
        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_error = 0.0
        self.prev_action = 0.0
        self.error_history = deque(maxlen=20)
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Compute steering action using optimized feedback structure.
        
        Args:
            target_lataccel: The target lateral acceleration.
            current_lataccel: The current lateral acceleration.
            state: The current state of the vehicle (SimpleNamespace with v_ego, a_ego, roll_lataccel).
            future_plan: The future plan for the next N frames (SimpleNamespace with lataccel array).
            
        Returns:
            The control signal (steer action) to be applied to the vehicle.
        """
        p = self.params
        
        # Extract future plan as array
        future_array = np.array(future_plan.lataccel) if hasattr(future_plan, 'lataccel') else []
        
        # Get lookahead target
        lookahead_idx = int(p['lookahead_idx'])
        if len(future_array) > lookahead_idx:
            lookahead_target = future_array[lookahead_idx]
        else:
            lookahead_target = target_lataccel
            
        # Velocity-dependent gain scaling
        v_ego = state.v_ego if hasattr(state, 'v_ego') else 20.0
        velocity_factor = 1.0 + p['velocity_scaling'] * (v_ego - 20.0)
        
        # Compute errors
        error_immediate = target_lataccel - current_lataccel
        
        # Filter the error to reduce noise sensitivity
        self.filtered_error = (
            p['error_filter_alpha'] * error_immediate +
            (1 - p['error_filter_alpha']) * self.filtered_error
        )
        self.error_history.append(error_immediate)
        
        # PID terms on filtered error
        p_term = p['kp'] * self.filtered_error * velocity_factor
        
        # Integral with anti-windup
        self.integral += self.filtered_error
        integral_limit = 2.0 / max(p['ki'], 0.001)
        self.integral = np.clip(self.integral, -integral_limit, integral_limit)
        i_term = p['ki'] * self.integral
        
        # Derivative using error history for robustness
        if len(self.error_history) >= 2:
            derivative = self.error_history[-1] - self.error_history[-2]
        else:
            derivative = 0.0
        d_term = p['kd'] * derivative
        
        # Feedforward terms
        roll_lataccel = state.roll_lataccel if hasattr(state, 'roll_lataccel') else 0.0
        roll_ff = -p['k_roll'] * roll_lataccel
        target_ff_immediate = p['k_ff_immediate'] * target_lataccel
        target_ff_lookahead = p['k_ff_lookahead'] * lookahead_target
        
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