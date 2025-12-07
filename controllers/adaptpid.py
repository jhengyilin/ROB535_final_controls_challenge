from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  An adaptive PID controller that adjusts gains based on vehicle state and error characteristics.
  
  The controller adapts PID gains based on:
  - Vehicle velocity (v_ego): Different dynamics at different speeds
  - Error magnitude: Larger errors may need more aggressive response
  - Error rate of change: Fast-changing errors may need different tuning
  - Current acceleration (a_ego): Different behavior during acceleration/deceleration
  """
  def __init__(self, 
               kp_base=0.25, ki_base=0.12, kd_base=-0.06,
               v_ego_scale=0.1, error_scale=0.2, accel_scale=0.05):
    # Base PID gains (starting point, similar to 2DOF PID)
    self.kp_base = kp_base
    self.ki_base = ki_base
    self.kd_base = kd_base
    
    # Adaptation scaling factors
    self.v_ego_scale = v_ego_scale  # How much velocity affects gains
    self.error_scale = error_scale   # How much error magnitude affects gains
    self.accel_scale = accel_scale  # How much acceleration affects gains
    
    # State variables
    self.error_integral = 0
    self.prev_error = 0
    self.prev_target = None
    self.prev_output = None
    
    # For adaptive gain calculation
    self.error_history = []  # Track recent errors for adaptation
    self.max_history = 10    # Keep last N errors
    
  def _adapt_gains(self, error, error_rate, v_ego, a_ego):
    """
    Adapt PID gains based on current conditions.
    
    Args:
      error: Current error (target - current)
      error_rate: Rate of change of error
      v_ego: Vehicle velocity
      a_ego: Vehicle acceleration
      
    Returns:
      Tuple of (kp, ki, kd) adapted gains
    """
    # Normalize velocity (assuming typical range 0-40 m/s)
    v_norm = np.clip(v_ego / 20.0, 0.0, 2.0)  # Normalize to [0, 2]
    
    # Normalize acceleration (assuming range -5 to 5 m/s^2)
    a_norm = np.clip(a_ego / 5.0, -1.0, 1.0)
    
    # Error magnitude adaptation: larger errors need more aggressive response
    error_mag = abs(error)
    error_mag_norm = np.clip(error_mag / 2.0, 0.0, 1.0)  # Normalize to [0, 1]
    
    # Error rate adaptation: fast-changing errors may need different tuning
    error_rate_mag = abs(error_rate)
    error_rate_norm = np.clip(error_rate_mag / 5.0, 0.0, 1.0)  # Normalize to [0, 1]
    
    # Adaptive proportional gain
    # - Increase with error magnitude (more aggressive for large errors)
    # - Slightly decrease with velocity (more stable at high speeds)
    # - Adjust based on acceleration state
    kp_mult = 1.0 + self.error_scale * error_mag_norm
    kp_mult -= 0.1 * self.v_ego_scale * v_norm  # Slightly reduce at high speeds
    kp_mult += 0.05 * self.accel_scale * abs(a_norm)  # Slightly increase during acceleration
    
    # Adaptive integral gain
    # - Increase with error magnitude (faster correction for large errors)
    # - Decrease with velocity (less integral action at high speeds to avoid overshoot)
    # - Reduce during acceleration to prevent windup
    ki_mult = 1.0 + 0.5 * self.error_scale * error_mag_norm
    ki_mult -= 0.2 * self.v_ego_scale * v_norm  # Reduce at high speeds
    ki_mult -= 0.1 * self.accel_scale * abs(a_norm)  # Reduce during acceleration
    
    # Adaptive derivative gain
    # - Increase with error rate (more damping for fast changes)
    # - Increase with velocity (more damping at high speeds)
    # - Adjust based on acceleration
    kd_mult = 1.0 + 0.3 * self.error_scale * error_rate_norm
    kd_mult += 0.15 * self.v_ego_scale * v_norm  # Increase at high speeds
    kd_mult += 0.05 * self.accel_scale * abs(a_norm)
    
    # Apply multipliers with bounds to prevent extreme values
    kp = self.kp_base * np.clip(kp_mult, 0.5, 2.0)
    ki = self.ki_base * np.clip(ki_mult, 0.3, 2.0)
    kd = self.kd_base * np.clip(kd_mult, 0.5, 2.0)
    
    return kp, ki, kd
  
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    """
    Update the controller with new measurements and compute control output.
    
    Args:
      target_lataccel: The target lateral acceleration (reference)
      current_lataccel: The current lateral acceleration (output)
      state: The current state of the vehicle (State namedtuple with roll_lataccel, v_ego, a_ego)
      future_plan: The future plan for the next N frames
    
    Returns:
      The control signal (steering command)
    """
    error = target_lataccel - current_lataccel
    
    # Track error history for adaptation
    self.error_history.append(error)
    if len(self.error_history) > self.max_history:
      self.error_history.pop(0)
    
    # Calculate error rate (derivative of error)
    if self.prev_error is not None:
      error_rate = error - self.prev_error
    else:
      error_rate = 0.0
    
    # Get vehicle state
    v_ego = state.v_ego
    a_ego = state.a_ego
    
    # Adapt gains based on current conditions
    kp, ki, kd = self._adapt_gains(error, error_rate, v_ego, a_ego)
    
    # Integral term: accumulate error with anti-windup
    # Limit integral accumulation to prevent windup
    max_integral = 10.0  # Maximum integral value
    self.error_integral += error
    self.error_integral = np.clip(self.error_integral, -max_integral, max_integral)
    
    # Proportional term
    proportional_term = kp * error
    
    # Integral term
    integral_term = ki * self.error_integral
    
    # Derivative term: use error derivative
    derivative_term = kd * error_rate
    
    # Update previous values
    self.prev_error = error
    self.prev_target = target_lataccel
    self.prev_output = current_lataccel
    
    # Compute control output
    control_output = proportional_term + integral_term + derivative_term
    
    return control_output

