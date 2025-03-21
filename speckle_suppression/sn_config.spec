[DM_REGISTRATION]
    calspot_kx = float
    calspot_ky = float
    calspot_amp = float(min=0)

[SN_SETTINGS]
    #x and y setpoints
    xcen = integer
    ycen = integer

[SIMULATION]
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
    
    [[CAMERA_PARAMS]]
        flux                 = float(min=0)
        exptime              = float(min=0)
        read_noise           = float(min=0)
        dark_current_rate    = float(min=0)
        flat_field           = float
        include_photon_noise = boolean
        xsize                = integer(min=0)
        ysize                = integer(min=0)
        field_center_x       = integer(min=0)
        field_center_y       = integer(min=0)
    
    [[AO_PARAMS]]
        modebasis               = option_or_none('zernike', 'pixel', 'disk_harmonics', 'fourier')
        initial_rms_wfe         = float(min=0)
        seed                    = integer
        rotation_angle_dm       = float
        num_actuators_across    = integer_or_none
        actuator_spacing        = float_or_none