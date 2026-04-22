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
            type                          = option('lyot', 'vortex')
            IWA_mas                       = float(min=0)
            charge                        = integer(min=1)
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
        bad_pixel_fraction   = float(min=0, max=1, default=0)
        output_directory     = string_or_none
        readout_delay        = float(min=0, default=0)

    [[AO_PARAMS]]
        modebasis               = option_or_none('zernike', 'pixel', 'disk_harmonics', 'fourier')
        initial_rms_wfe         = float(min=0)
        seed                    = integer_or_none
        rotation_angle_dm       = float
        flip_x_dm               = boolean
        flip_y_dm               = boolean
        num_actuators_across    = integer_or_none
        actuator_spacing        = float_or_none
