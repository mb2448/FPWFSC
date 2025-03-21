[SN_SETTINGS]
    #x and y setpoints
    xcen = integer
    ycen = integer

[SIMULATION]
    flux            = float(min=0)
    exptime         = float(min=0)
    rms_wfe         = float(min=0)
    seed            = integer

    wavelength (m)                = float(min=0, max=10e-6)
    pixel scale (mas/pix)         = float(min=0)
    N pix pupil                   = integer
    N pix focal                   = integer
    aperture                      = option('open', 'subaru', 'keck')
    lyot stop                     = option('open','NIRC2_large_hexagonal_mask', 'NIRC2_incircle_mask','NIRC2_Lyot_Stop', 'keck+NIRC2_Lyot_Stop')
    coronagraph IWA (mas)         = float(min=0)
    rotation angle aperture (deg) = float
    rotation angle dm (deg)       = float
    rotation angle im (deg)       = float
    flip_x                        = boolean
    flip_y                        = boolean
    ref PSF oversampling factor   = integer(min=1)