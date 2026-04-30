[EXECUTION]
    plot = boolean(default=True)
    log = boolean(default=False)
    logdir = string(default='.')
    N iterations = integer(min=1, default=100)
    x setpoint = float(default=512.0)
    y setpoint = float(default=512.0)
    inner radius = integer(min=0, default=10)
    outer radius = integer(min=3, default=100)

[PID]
    x centroid offset = float(min=-1, max=1, default=0)
    y centroid offset = float(min=-1, max=1, default=0)
    Kp = float(min=-1, max=1, default=0.5)
    Ki = float(min=-1, max=1, default=0.1)
    Kd = float(min=-1, max=1, default=0)
    output_limits = float(default=0.5)

[CAMERA CALIBRATION]
    background file = string(default='')
    masterflat file = string(default='')
    badpix file = string(default='')

[HITCHHIKER MODE]
    hitchhike = boolean(default=False)
    poll interval = float(min=0.001, default=0.5)
    timeout = float(min=0.001, default=20)
    imagedir = string(default='/')

[AO]
    tip tilt gain = float(default=-250e-9)
    tip tilt angle (deg)= float(min=-360, max=360, default=0)
    tip tilt flip x = boolean(default=False)
    tip tilt flip y = boolean(default=False)
