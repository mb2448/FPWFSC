[SN_SETTINGS]
    #x and y setpoints
    xcen = integer
    ycen = integer
    cropsize = integer_or_none

[DM_REGISTRATION]
    calspot_kx = float
    calspot_ky = float
    calspot_amp = float(min=0)


[SIMULATION]
    [[OPTICAL_PARAMS]]
        wavelength (m)                        = float(min=0, max=10e-6)
        N pix pupil                       = integer
        N pix focal                       = integer
        pixel scale (mas/pix)             = float(min=0)
        rotation angle im (deg)           = float(min=0)
        #flips are applied last, not first
        flip_x                             = boolean
        flip_y                             = boolean
        [[[APERTURE]]]
            aperture                      = option('keck', 'subaru')
            rotation angle aperture (deg) = float
        [[[CORONAGRAPH_MASK]]]
            IWA_mas                       = float(min=0)
        [[[LYOT_STOP]]]
            lyot stop                     = option('NIRC2_incircle_mask', 'NIRC2_large_hexagonal_mask', 'NIRC2_Lyot_Stop')
            rotation angle lyot (deg)     = float
    
    
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
        seed                    = integer_or_none
        rotation_angle_dm       = float
        num_actuators_across    = integer_or_none
        actuator_spacing        = float_or_none