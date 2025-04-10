#!/usr/bin/env python
import sys
import threading
import numpy as np
from collections import deque
import hcipy
import support_functions as sf
import plotting_funcs as pf
from configobj import ConfigObj
import classes as ff_c
import fake_hardware as fhw

import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/mbottom/Desktop/Useful_code_fragments/plotting_and_animation')
from ds9 import ds9
# to make sure we always have the same aberration

def run(my_deque=None, my_event=None):

    if my_deque is None:
        my_deque = deque()

    if my_event is None:
        my_event = threading.Event()

    FF_ini  = 'FF_software_sim.ini'
    FF_spec = 'FF_software.spec'
    settings = sf.validate_config(FF_ini, FF_spec)

    #----------------------------------------------------------------------
    # Control Loop parameters
    #----------------------------------------------------------------------
    Niter             = settings['LOOP_SETTINGS']['N iter']
    gain              = settings['LOOP_SETTINGS']['gain']
    leak_factor       = settings['LOOP_SETTINGS']['leak factor']
    chosen_mode_basis = settings['LOOP_SETTINGS']['Used mode basis']
    Nmodes            = settings['LOOP_SETTINGS']['Number of modes']
    Nimg_avg          = settings['LOOP_SETTINGS']['N images averaged']
    control_even      = settings['LOOP_SETTINGS']['control even modes']
    control_odd       = settings['LOOP_SETTINGS']['control odd modes']
    dm_command_boost  = settings['LOOP_SETTINGS']['dm command boost']
    
    #----------------------------------------------------------------------
    # Optical model parameters
    #----------------------------------------------------------------------
    # Optical properties
    wavelength = settings['MODELLING']['wavelength (m)']
    mas_pix = settings['MODELLING']['pixel scale (mas/pix)']
    
    # Pupil and focal plane sampling
    Npix_pup = settings['MODELLING']['N pix pupil']
    Npix_foc = settings['MODELLING']['N pix focal']

    # Aperture and DM configuration
    chosen_aperture = settings['MODELLING']['aperture']
    rotation_angle_aperture = settings['MODELLING']['rotation angle aperture (deg)']
    rotation_angle_dm = settings['MODELLING']['rotation angle dm (deg)']
    
    # Image orientation settings
    rotation_angle_deg = settings['MODELLING']['rotation angle im (deg)']
    flip_x = settings['MODELLING']['flip_x']
    flip_y = settings['MODELLING']['flip_y']
    
    # Reference PSF settings
    oversampling_factor = settings['MODELLING']['ref PSF oversampling factor']
    #----------------------------------------------------------------------
    # F&F parameters
    #----------------------------------------------------------------------
    xcen                = settings['FF_SETTINGS']['xcen']
    ycen                = settings['FF_SETTINGS']['ycen']
    #WILBY SMOOTHIN IS NOT IMPLEMENTED YET!!!
    apply_smooth_filter = settings['FF_SETTINGS']['Apply smooth filter']
    epsilon             = settings['FF_SETTINGS']['epsilon']
    SNR_cutoff          = settings['FF_SETTINGS']['SNR cutoff']
    #automatically sub the background in the camera frame
    auto_background     = settings['FF_SETTINGS']['auto_background']   

    #----------------------------------------------------------------------
    # Load the classes
    #----------------------------------------------------------------------
    Aperture = ff_c.Aperture(Npix_pup=Npix_pup, 
                             aperturename=chosen_aperture,
                             rotation_angle_aperture=rotation_angle_aperture)

    #note flux does not matter unless you are doing a sim
    OpticalModel = ff_c.SystemModel(aperture=Aperture,
                                    Npix_foc=Npix_foc,
                                    mas_pix=mas_pix,
                                    wavelength=wavelength)

    FnF = ff_c.FastandFurious(SystemModel=OpticalModel,
                              leak_factor=leak_factor,
                              gain=gain,
                              epsilon=epsilon,
                              chosen_mode_basis=chosen_mode_basis,
                              #apply_smoothing_filter=apply_smooth_filter,
                              number_of_modes=Nmodes)
    #----------------------------------------------------------------------
    # Load instruments
    #----------------------------------------------------------------------
    Camera = fhw.FakeDetector(opticalsystem=OpticalModel,
                                   flux = 1e6,
                                   exptime=1,
                                   dark_current_rate=0,
                                   read_noise=10,
                                   flat_field=0,
                                   include_photon_noise=True,
                                   xsize=1024,
                                   ysize=1024,
                                   field_center_x=333,
                                   field_center_y=433)
    
    AOsystem = fhw.FakeAOSystem(OpticalModel, modebasis=FnF.mode_basis, initial_rms_wfe=0.75)
    
    # generating the first reference image
    data_raw = Camera.take_image()
    data_ref = sf.reduce_images(data_raw, xcen=xcen, ycen=ycen, 
                                          npix=Npix_foc, 
                                          flipx = flip_x, flipy=flip_y, 
                                          refpsf=OpticalModel.ref_psf.shaped, 
                                          rotation_angle_deg=rotation_angle_deg)
    # Take first image
    FnF.initialize_first_image(data_ref)
    #MAIN LOOP SETUP AND RUNNING
    #----------------------------------------------------------------------

    RMS_measurements = np.zeros(Niter)
    SRA_measurements = np.zeros(Niter)
    VAR_measurements = np.zeros(Niter)

    RMS_measurements[RMS_measurements==0] = np.nan
    SRA_measurements[SRA_measurements==0] = np.nan
    VAR_measurements[VAR_measurements==0] = np.nan

    for i in np.arange(Niter):
        if my_event.is_set():
            return

        SRA_measurements[i] = FnF.estimate_strehl()
        my_deque.append(SRA_measurements)

        img = Camera.take_image()
        data = sf.reduce_images(img, xcen=xcen, ycen=ycen, npix=Npix_foc, 
                                flipx = flip_x, flipy=flip_y, 
                                refpsf=OpticalModel.ref_psf.shaped, 
                                rotation_angle_deg=rotation_angle_deg)
        #update the loop with the new data
        phase_DM = FnF.iterate(data)
        microns = phase_DM * FnF.wavelength / (2 * np.pi) * 1e6
        dm_microns = AOsystem.make_dm_command(microns)
        AOsystem.set_dm_data(dm_microns)
        
        # Saving metrics of strehl, airy ring variation, and rmms
        VAR_measurements[i] = sf.calculate_VAR(data, OpticalModel.ref_psf.shaped,
                                               mas_pix, wavelength,
                                               Aperture.pupil_diameter)
        #-------------------------------------------
        # plotting
        #-------------------------------------------
        #pf.plot_progress(Niter=Niter,
        #                 data=FnF.previous_image,
        #                 pupil_wf=phase_DM,
        #                 aperture=FnF.aperture,
        #                 SRA=SRA_measurements,
        #                 VAR=VAR_measurements,
        #                 RMS=RMS_measurements)

if __name__ == "__main__":
    run()