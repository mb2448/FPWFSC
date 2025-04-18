[HITCHHIKER MODE]
    hitchhike = boolean(default=False)
    imagedir = string(default='/')

[EXECUTION]
    plot = boolean(default=True)
    n_iter = integer(min=1, default=100)
    xcen = float(default=512.0)
    ycen = float(default=512.0)
    #radius in pixels to search for spots
    spot search radius (pix) = float(min=0, default=60)
    radius tolerance (pix) = float(min=0, default=20)

[AO]
    default_waffle_amplitude = float
    tt_gain     = float(default=-250e-9)
    tt_rot_deg  = float(min=-360, max=360, default=0)
    tt_flipx    = boolean(default=False)
    tt_flipy    = boolean(default=False)

[PID]
    Kp = float(min=0, default=0.5)
    Ki = float(min=0, default=0.1)
    Kd = float(min=0, default=0)
    output_limits = float(default=3)
