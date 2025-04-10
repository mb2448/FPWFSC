def _clamp(value, limits):
    lower, upper = limits
    if value is None:
        return None
    
    # Handle scalar values
    if not hasattr(value, '__iter__') or isinstance(value, (str, bytes)):
        if (upper is not None) and (value > upper):
            return upper
        elif (lower is not None) and (value < lower):
            return lower
        return value
    
    # Handle vector/list/array inputs by making a copy and modifying it
    result = value.copy() if hasattr(value, 'copy') else list(value)
    
    for i in range(len(result)):
        if (upper is not None) and (result[i] > upper):
            result[i] = upper
        elif (lower is not None) and (result[i] < lower):
            result[i] = lower
    
    return result

class PID(object):
    """A simple PID controller."""

    def __init__(
        self,
        Kp=1.0,
        Ki=0.0,
        Kd=0.0,
        setpoint=0,
        output_limits=(None, None),
        auto_mode=True,
        proportional_on_measurement=False,
        differential_on_measurement=True,
        error_map=None,
        starting_output=0.0,
    ):
        """
        Initialize a new PID controller.

        :param Kp: The value for the proportional gain Kp
        :param Ki: The value for the integral gain Ki
        :param Kd: The value for the derivative gain Kd
        :param setpoint: The initial setpoint that the PID will try to achieve
        :param output_limits: The initial output limits to use, given as an iterable with 2
            elements, for example: (lower, upper). The output will never go below the lower limit
            or above the upper limit. Either of the limits can also be set to None to have no limit
            in that direction. Setting output limits also avoids integral windup, since the
            integral term will never be allowed to grow outside of the limits.
        :param auto_mode: Whether the controller should be enabled (auto mode) or not (manual mode)
        :param proportional_on_measurement: Whether the proportional term should be calculated on
            the input directly rather than on the error (which is the traditional way). Using
            proportional-on-measurement avoids overshoot for some types of systems.
        :param differential_on_measurement: Whether the differential term should be calculated on
            the input directly rather than on the error (which is the traditional way).
        :param error_map: Function to transform the error value in another constrained value.
        :param starting_output: The starting point for the PID's output. If you start controlling
            a system that is already at the setpoint, you can set this to your best guess at what
            output the PID should give when first calling it to avoid the PID outputting zero and
            moving the system away from the setpoint.
        """
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint

        self._min_output, self._max_output = None, None
        self._auto_mode = auto_mode
        self.proportional_on_measurement = proportional_on_measurement
        self.differential_on_measurement = differential_on_measurement
        self.error_map = error_map

        self._proportional = 0
        self._integral = 0
        self._derivative = 0

        self._last_output = None
        self._last_error = None
        self._last_input = None

        self.output_limits = output_limits
        self.reset()

        # Set initial state of the controller
        self._integral = _clamp(starting_output, output_limits)

    def __call__(self, input_):
        """
        Update the PID controller.

        Call the PID controller with *input_* and calculate and return a control output.
        Assumes regular sampling interval.

        """
        if not self.auto_mode:
            return self._last_output

        # Handle vector inputs
        is_vector_input = hasattr(input_, '__iter__') and not isinstance(input_, (str, bytes))
        
        # Compute error terms
        if is_vector_input:
            # Vector operations
            error = tuple(sp - inp for sp, inp in zip(self.setpoint if hasattr(self.setpoint, '__iter__') else [self.setpoint]*len(input_), input_))
            
            if self._last_input is None:
                d_input = tuple(0 for _ in input_)
            else:
                d_input = tuple(curr - prev for curr, prev in zip(input_, self._last_input))
                
            if self._last_error is None:
                d_error = tuple(0 for _ in error)
            else:
                d_error = tuple(curr - prev for curr, prev in zip(error, self._last_error))
        else:
            # Scalar operations (original code)
            error = self.setpoint - input_
            d_input = input_ - (self._last_input if (self._last_input is not None) else input_)
            d_error = error - (self._last_error if (self._last_error is not None) else error)

        # Check if must map the error
        if self.error_map is not None:
            error = self.error_map(error)

        # Compute the proportional term
        if not self.proportional_on_measurement:
            # Regular proportional-on-error, simply set the proportional term
            if is_vector_input:
                self._proportional = tuple(self.Kp * e for e in error)
            else:
                self._proportional = self.Kp * error
        else:
            # Add the proportional error on measurement to error_sum
            if is_vector_input:
                if not hasattr(self._proportional, '__iter__'):
                    self._proportional = tuple(0 for _ in input_)
                self._proportional = tuple(p - self.Kp * d for p, d in zip(self._proportional, d_input))
            else:
                self._proportional -= self.Kp * d_input

        # Compute integral and derivative terms
        if is_vector_input:
            if not hasattr(self._integral, '__iter__'):
                self._integral = tuple(0 for _ in input_)
            self._integral = tuple(i + self.Ki * e for i, e in zip(self._integral, error))
            self._integral = _clamp(self._integral, self.output_limits)  # Avoid integral windup

            if self.differential_on_measurement:
                self._derivative = tuple(-self.Kd * d for d in d_input)
            else:
                self._derivative = tuple(self.Kd * d for d in d_error)
        else:
            self._integral += self.Ki * error
            self._integral = _clamp(self._integral, self.output_limits)  # Avoid integral windup

            if self.differential_on_measurement:
                self._derivative = -self.Kd * d_input
            else:
                self._derivative = self.Kd * d_error

        # Compute final output
        if is_vector_input:
            output = tuple(p + i + d for p, i, d in zip(self._proportional, self._integral, self._derivative))
        else:
            output = self._proportional + self._integral + self._derivative
        
        output = _clamp(output, self.output_limits)

        # Keep track of state
        self._last_output = output
        self._last_input = input_
        self._last_error = error

        return output

    def __repr__(self):
        return (
            '{self.__class__.__name__}('
            'Kp={self.Kp!r}, Ki={self.Ki!r}, Kd={self.Kd!r}, '
            'setpoint={self.setpoint!r}, '
            'output_limits={self.output_limits!r}, auto_mode={self.auto_mode!r}, '
            'proportional_on_measurement={self.proportional_on_measurement!r}, '
            'differential_on_measurement={self.differential_on_measurement!r}, '
            'error_map={self.error_map!r}'
            ')'
        ).format(self=self)

    @property
    def components(self):
        """
        The P-, I- and D-terms from the last computation as separate components as a tuple. Useful
        for visualizing what the controller is doing or when tuning hard-to-tune systems.
        """
        return self._proportional, self._integral, self._derivative

    @property
    def tunings(self):
        """The tunings used by the controller as a tuple: (Kp, Ki, Kd)."""
        return self.Kp, self.Ki, self.Kd

    @tunings.setter
    def tunings(self, tunings):
        """Set the PID tunings."""
        self.Kp, self.Ki, self.Kd = tunings

    @property
    def auto_mode(self):
        """Whether the controller is currently enabled (in auto mode) or not."""
        return self._auto_mode

    @auto_mode.setter
    def auto_mode(self, enabled):
        """Enable or disable the PID controller."""
        self.set_auto_mode(enabled)

    def set_auto_mode(self, enabled, last_output=None):
        """
        Enable or disable the PID controller, optionally setting the last output value.

        This is useful if some system has been manually controlled and if the PID should take over.
        In that case, disable the PID by setting auto mode to False and later when the PID should
        be turned back on, pass the last output variable (the control variable) and it will be set
        as the starting I-term when the PID is set to auto mode.

        :param enabled: Whether auto mode should be enabled, True or False
        :param last_output: The last output, or the control variable, that the PID should start
            from when going from manual mode to auto mode. Has no effect if the PID is already in
            auto mode.
        """
        if enabled and not self._auto_mode:
            # Switching from manual mode to auto, reset
            self.reset()

            self._integral = last_output if (last_output is not None) else 0
            self._integral = _clamp(self._integral, self.output_limits)

        self._auto_mode = enabled

    @property
    def output_limits(self):
        """
        The current output limits as a 2-tuple: (lower, upper).

        See also the *output_limits* parameter in :meth:`PID.__init__`.
        """
        return self._min_output, self._max_output

    @output_limits.setter
    def output_limits(self, limits):
        """Set the output limits."""
        if limits is None:
            self._min_output, self._max_output = None, None
            return

        min_output, max_output = limits

        if (None not in limits) and (max_output < min_output):
            raise ValueError('lower limit must be less than upper limit')

        self._min_output = min_output
        self._max_output = max_output

        self._integral = _clamp(self._integral, self.output_limits)
        self._last_output = _clamp(self._last_output, self.output_limits)

    def reset(self):
        """
        Reset the PID controller internals.

        This sets each term to 0 as well as clearing the integral, the last output and the last
        input (derivative calculation).
        """
        self._proportional = 0
        self._integral = 0
        self._derivative = 0

        self._integral = _clamp(self._integral, self.output_limits)

        self._last_output = None
        self._last_input = None
        self._last_error = None