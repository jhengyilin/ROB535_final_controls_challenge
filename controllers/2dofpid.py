from . import BaseController

class Controller(BaseController):
  """
  A 2 Degree of Freedom PID controller
  
  This controller has separate tuning parameters for:
  - Setpoint tracking (b, c parameters)
  - Disturbance rejection (Kp, Ki, Kd gains)
  
  The control law is:
  u = Kp * (b*r - y) + Ki * ∫(r - y)dt + Kd * (c*dr/dt - dy/dt)
  
  where:
  - r = target_lataccel (reference)
  - y = current_lataccel (output)
  - b = setpoint weight for proportional term (0 ≤ b ≤ 1)
  - c = setpoint weight for derivative term (0 ≤ c ≤ 1)
  """
  def __init__(self, kp=0.25, ki=0.12, kd=-0.06, b=0.8, c=0.5):
    # PID gains
    self.kp = kp
    self.ki = ki
    self.kd = kd
    
    # 2DOF setpoint weights
    self.b = b  # Proportional setpoint weight (lower = less aggressive setpoint tracking)
    self.c = c  # Derivative setpoint weight (lower = less aggressive setpoint tracking)
    
    # State variables
    self.error_integral = 0
    self.prev_error = 0
    self.prev_target = None
    self.prev_output = None
    
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
    error = target_lataccel - current_lataccel
    
    # Integral term: accumulate error
    self.error_integral += error
    
    # Proportional term: 2DOF with setpoint weight b
    # Standard PID: Kp * error = Kp * (r - y)
    # 2DOF PID: Kp * (b*r - y) = Kp * (b*r - y) = Kp * (b*r - (r - error)) = Kp * ((b-1)*r + error)
    # But more directly: Kp * (b * target - current)
    proportional_term = self.kp * (self.b * target_lataccel - current_lataccel)
    
    # Integral term: standard (no setpoint weight needed)
    integral_term = self.ki * self.error_integral
    
    # Derivative term: 2DOF with setpoint weight c
    # Standard PID: Kd * d(error)/dt = Kd * d(r - y)/dt
    # 2DOF PID: Kd * (c*dr/dt - dy/dt)
    if self.prev_target is not None and self.prev_output is not None:
      # Approximate derivatives using backward difference
      target_derivative = target_lataccel - self.prev_target
      output_derivative = current_lataccel - self.prev_output
      derivative_term = self.kd * (self.c * target_derivative - output_derivative)
    else:
      # First step: use error derivative (fallback to standard PID)
      error_derivative = error - self.prev_error
      derivative_term = self.kd * error_derivative
    
    # Update previous values
    self.prev_error = error
    self.prev_target = target_lataccel
    self.prev_output = current_lataccel
    
    # Compute control output
    control_output = proportional_term + integral_term + derivative_term
    
    return control_output

