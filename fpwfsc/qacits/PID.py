#
#Modified from
#https://github.com/m-lundberg/simple-pid/blob/master/simple_pid/pid.py

import numpy as np

def _clamp(value, limits):
    """Clamp value(s) to specified limits."""
    if value is None or limits is None:
        return value

    # Handle single value limits (apply as +/- to all components)
    if np.isscalar(limits):
        lower, upper = -abs(limits), abs(limits)
    else:
        lower, upper = limits

    result = np.array(value, copy=True)

    if upper is not None:
        result = np.minimum(result, upper)
    if lower is not None:
        result = np.maximum(result, lower)

    return result

class PID:
    """Concise PID controller with numpy array support."""

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, setpoint=0.0, output_limits=None):
        """
        Initialize PID controller.

        Parameters:
        -----------
        Kp : float
            Proportional gain
        Ki : float
            Integral gain
        Kd : float
            Derivative gain
        setpoint : float or array-like
            Target setpoint(s)
        output_limits : float, tuple, or None
            Output limits. Can be:
            - Single value: applies +/- limit to all components
            - Tuple (lower, upper): asymmetric limits
            - None: no limits
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = np.asarray(setpoint)
        self.output_limits = output_limits

        # Internal state
        self._integral = 0.0
        self._last_input = None

    def iterate(self, current_position):
        """
        Perform one PID iteration.

        Parameters:
        -----------
        current_position : float or array-like
            Current measured position(s)

        Returns:
        --------
        output : float or ndarray
            PID control output
        """
        # Convert to numpy array
        position = np.asarray(current_position)

        # Ensure setpoint matches input dimensions
        setpoint = self.setpoint
        if position.ndim > 0 and setpoint.ndim == 0:
            setpoint = np.full_like(position, setpoint)

        # Compute error
        error = setpoint - position

        # Proportional term
        proportional = self.Kp * error

        # Integral term with anti-windup
        if self._integral is None or (position.ndim > 0 and np.isscalar(self._integral)):
            self._integral = np.zeros_like(position)

        self._integral += self.Ki * error
        self._integral = _clamp(self._integral, self.output_limits)

        # Derivative term
        if self._last_input is None:
            derivative = np.zeros_like(position)
        else:
            d_input = position - self._last_input
            derivative = -self.Kd * d_input  # Derivative on measurement to avoid spikes

        # Compute output
        output = proportional + self._integral + derivative
        output = _clamp(output, self.output_limits)

        # Update state
        self._last_input = position

        return output

    def update_setpoint(self, new_setpoint):
        """
        Update the PID setpoint.

        Parameters:
        -----------
        new_setpoint : float or array-like
            New target setpoint(s)
        """
        self.setpoint = np.asarray(new_setpoint)

    def reset(self):
        """Reset PID internal state."""
        self._integral = 0.0
        self._last_input = None

    @property
    def gains(self):
        """Get PID gains as tuple (Kp, Ki, Kd)."""
        return self.Kp, self.Ki, self.Kd

    @gains.setter
    def gains(self, values):
        """Set PID gains from tuple (Kp, Ki, Kd)."""
        self.Kp, self.Ki, self.Kd = values

    def __repr__(self):
        return (f"PID(Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}, "
                f"setpoint={self.setpoint}, output_limits={self.output_limits})")


# Example usage
if __name__ == "__main__":
    print("=== Scalar PID Test ===")
    pid_scalar = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=10.0, output_limits=5)  # ±5 limit

    position = 8.0
    for i in range(6):
        output = pid_scalar.iterate(position)
        position += output * 0.1  # Simulate system response
        print(f"Step {i}: position={position:.3f}, output={output:.3f}")

    print("\n=== Vector PID Test (x,y control) ===")
    pid_vector = PID(Kp=0.8, Ki=0.05, Kd=0.1,
                    setpoint=np.array([0.0, 0.0]),
                    output_limits=2)  # ±2 limit for both x and y

    position = np.array([1.5, -0.8])  # Starting offset
    for i in range(6):
        output = pid_vector.iterate(position)
        position += output * 0.2  # Simulate system response
        print(f"Step {i}: position={position}, output={output}")

    print("\n=== Setpoint Change Test ===")
    pid_vector.update_setpoint(np.array([2.0, -1.0]))
    for i in range(3):
        output = pid_vector.iterate(position)
        position += output * 0.2
        print(f"New setpoint step {i}: position={position}, output={output}")

    print(f"\nFinal PID state: {pid_vector}")
