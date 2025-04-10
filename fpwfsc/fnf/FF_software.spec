[LOOP_SETTINGS]
    Plot               = boolean
    N iter             = integer(min=1)
    gain               = float(min=0)
    leak factor        = float(min=0, max=1)
    Used mode basis    = option('zernike', 'pixel', 'disk_harmonics', 'fourier')
    Number of modes    = integer(min=1)
    N images averaged  = integer(min=1)
    control even modes = boolean
    control odd modes  = boolean

[MODELLING]
    wavelength (m)                = float(min=0, max=10e-6)
    pixel scale (mas/pix)         = float(min=0)
    N pix pupil                   = integer
    N pix focal                   = integer
    aperture                      = option('open', 'subaru', 'keck', 'NIRC2_large_hexagonal_mask', 'NIRC2_incircle_mask', 'keck+NIRC2_large_hexagonal_mask', 'keck+NIRC2_incircle_mask', 'NIRC2_Lyot_Stop', 'keck+NIRC2_Lyot_Stop')
    rotation angle aperture (deg) = float
    rotation angle dm (deg)       = float
    rotation angle im (deg)       = float
    flip_x                        = boolean
    flip_y                        = boolean
    ref PSF oversampling factor   = integer(min=1)

[FF_SETTINGS]
    xcen                = integer
    ycen                = integer
    Apply smooth filter = boolean
    SNR cutoff          = float
    epsilon             = float
    auto_background     = boolean

[IO]
    save path       = string
    plot dm command = boolean

[SIMULATION]
    flux            = float(min=0)
    exptime         = float(min=0)
    rms_wfe         = float(min=0)
    seed            = integer