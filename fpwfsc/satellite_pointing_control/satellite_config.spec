[HITCHHIKER MODE]
    hitchhike = boolean
    imagedir = string

[SETPOINT]
    xcen = float
    ycen = float
    #radius in pixels to search for spots
    spot search radius (pix) = float(min=0)
    radius tolerance (pix) = float(min=0)

[SPOT SETTINGS]
    amplitude = float

[PID]
    Kp = float(min=0)
    Ki = float(min=0)
    Kd = float(min=0)
    output_limits = float
