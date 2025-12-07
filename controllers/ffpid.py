# Average lataccel_cost:  1.064, average jerk_cost:  25.94, average total_cost:  79.14

from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A Preview-based Feedforward + Feedback PID controller
  
  This controller uses:
  - Feedforward: Estimates required control from future plan to anticipate upcoming changes
  - Feedback: PID controller to correct tracking errors
  - Smoothing: Low-pass filtering to reduce jerk
  
  The feedforward term uses the future target lateral acceleration to predict
  the required steering command, while the feedback PID corrects for any
  tracking errors and disturbances.
  """
  def __init__(self, kp=0.25, ki=0.13, kd=-0.05, ff_gain=1.5, smooth_alpha=0.1):
    # PID gains for feedback
    self.kp = kp
    self.ki = ki
    self.kd = kd
    
    # Feedforward gain (how much to trust the preview)
    self.ff_gain = ff_gain
    
    # Smoothing factor for output (0-1, higher = more smoothing, less jerk)
    self.smooth_alpha = smooth_alpha
    
    # State variables
    self.error_integral = 0
    self.prev_error = 0
    self.prev_output = None
    
    # For estimating feedforward from future plan
    # Simple model: steer ≈ (target_lataccel - roll_lataccel) / (v_ego * gain_factor)
    self.ff_velocity_factor = 0.03  # Estimated gain from velocity
    
  def _compute_feedforward(self, target_lataccel, state, future_plan):
    """
    Compute feedforward control from current and future targets.
    
    Uses a simple inverse model: if we know the target lateral acceleration,
    we can estimate the required steering command.
    """
    if future_plan is None or len(future_plan.lataccel) == 0:
      # No future plan available, use current target only
      net_target = target_lataccel - state.roll_lataccel
      # Simple inverse model: steer proportional to desired lateral accel / velocity
      v_ego = max(state.v_ego, 1.0)  # Avoid division by zero
      feedforward = net_target * self.ff_velocity_factor / v_ego
      return feedforward
    
    # Use preview: look ahead a few steps to anticipate changes
    preview_horizon = min(5, len(future_plan.lataccel))
    
    # Average of near-future targets (weighted more on immediate future)
    weights = np.exp(-np.arange(preview_horizon) * 0.3)  # Exponential decay
    weights = weights / np.sum(weights)
    
    weighted_target = 0.0
    for i in range(preview_horizon):
      # Account for roll in future plan
      net_target = future_plan.lataccel[i] - future_plan.roll_lataccel[i]
      weighted_target += weights[i] * net_target
    
    # Also include current target with some weight
    current_net = target_lataccel - state.roll_lataccel
    weighted_target = 0.6 * current_net + 0.4 * weighted_target
    
    # Inverse model: estimate steer from desired lateral acceleration
    v_ego = max(state.v_ego, 1.0)
    feedforward = weighted_target * self.ff_velocity_factor / v_ego
    
    return feedforward
  
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    """
    Update the controller with new measurements
    
    Args:
      target_lataccel: The target lateral acceleration (reference)
      current_lataccel: The current lateral acceleration (output)
      state: The current state of the vehicle
      future_plan: The future plan for the next N frames
    
    Returns:
      The control signal (steering command)
    """
    # Compute error
    error = target_lataccel - current_lataccel
    
    # Integral term: accumulate error with anti-windup
    self.error_integral += error
    max_integral = 10.0  # Limit integral to prevent windup
    self.error_integral = np.clip(self.error_integral, -max_integral, max_integral)
    
    # Feedback PID terms
    proportional_term = self.kp * error
    integral_term = self.ki * self.error_integral
    error_derivative = error - self.prev_error
    derivative_term = self.kd * error_derivative
    
    feedback = proportional_term + integral_term + derivative_term
    
    # Feedforward: anticipate from future plan
    feedforward = self._compute_feedforward(target_lataccel, state, future_plan)
    
    # Combine feedforward and feedback
    # Feedforward anticipates, feedback corrects errors
    control_output = self.ff_gain * feedforward + feedback
    
    # Smooth the output to reduce jerk
    if self.prev_output is not None:
      # Low-pass filter: smooth_alpha * previous + (1 - smooth_alpha) * current
      control_output = self.smooth_alpha * self.prev_output + (1.0 - self.smooth_alpha) * control_output
    
    # Update previous values
    self.prev_error = error
    self.prev_output = control_output
    
    return control_output

