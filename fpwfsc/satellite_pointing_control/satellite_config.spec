[HITCHHIKER MODE]
    hitchhike = boolean(default=False)
    poll interval = float(min=0.001, default=0.5)
    timeout = float(min=0.001, default=20)
    imagedir = string(default='/')

[EXECUTION]
    plot = boolean(default=True)
    N iterations = integer(min=1, default=100)
    x setpoint = float(default=512.0)
    y setpoint = float(default=512.0)
    #radius in pixels to search for spots
    spot search radius (pix) = float(min=0, default=60)
    radius tolerance (pix) = float(min=0, default=20)

[AO]
    waffle mode amplitude = float(default = 150e-9)
    tip tilt gain = float(default=-250e-9)
    tip tilt angle (deg)= float(min=-360, max=360, default=0)
    tip tilt flip x = boolean(default=False)
    tip tilt flip y = boolean(default=False)

[PID]
    Kp = float(min=0, default=0.5)
    Ki = float(min=0, default=0.1)
    Kd = float(min=0, default=0)
    output_limits = float(default=3)
