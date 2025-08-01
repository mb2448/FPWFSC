#!/usr/bin/env python
import numpy as np
from collections import deque
import hcipy
from configobj import ConfigObj
import time
import matplotlib.pyplot as plt
from pathlib import Path

from fpwfsc.fnf import gui_helper as helper

from fpwfsc.common import plotting_funcs as pf
from fpwfsc.common import classes as ff_c
from fpwfsc.common import fake_hardware as fhw
from fpwfsc.common import support_functions as sf
import random

def run_fastandfurious_test(camera='Sim', 
                            aosystem='Sim', 
                            FF_ini = 'FF_software_sim.ini', 
                            FF_spec = 'FF_software.spec'):

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
    #this is not used right now anywhere in this script
    auto_background     = settings['FF_SETTINGS']['auto_background']

    #----------------------------------------------------------------------
    # Simulation parameters
    #----------------------------------------------------------------------
    flux                = settings['SIMULATION']['flux']
    exptime             = settings['SIMULATION']['exptime']
    rms_wfe             = settings['SIMULATION']['rms_wfe']
    seed                = settings['SIMULATION']['seed']
    #----------------------------------------------------------------------
    # Load the classes
    #----------------------------------------------------------------------
   


    Aperture = ff_c.Aperture(Npix_pup=Npix_pup,
                             aperturename=chosen_aperture,
                             rotation_angle_aperture=rotation_angle_aperture)

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
    if camera == 'Sim' and aosystem == 'Sim':
        Camera = fhw.FakeDetector(opticalsystem=OpticalModel,
                                  flux = flux,
                                  exptime=exptime,
                                  dark_current_rate=0,
                                  read_noise=10,
                                  flat_field=0,
                                  include_photon_noise=True,
                                  xsize=1024,
                                  ysize=1024,
                                  field_center_x=330,
                                  field_center_y=430)

        AOsystem = fhw.FakeAOSystem(OpticalModel, modebasis=FnF.mode_basis,
                                    initial_rms_wfe=rms_wfe, seed=seed,
                                    rotation_angle_dm = rotation_angle_dm,
                                    flip_x = flip_x,
                                    flip_y  = flip_y)
    else:
        Camera = camera
        AOsystem = aosystem
    

    #create zernike mode
    mode_basis = hcipy.make_zernike_basis(10, 11.3, Aperture.pupil_grid, 1)
    mode_basis = sf.orthonormalize_mode_basis(mode_basis, Aperture.aperture)
    amplitude = 0.3



    add_mode = amplitude * (
    mode_basis[0] * random.random() + mode_basis[1] * random.random() + mode_basis[2] * random.random() +
    mode_basis[3] * random.random() + mode_basis[4] * random.random() + mode_basis[5] * random.random() +
    mode_basis[6] * random.random() + mode_basis[7] * random.random() + mode_basis[8] * random.random() +
    mode_basis[9] * random.random()
)
    microns = add_mode * FnF.wavelength / (2 * np.pi) * 1e6
    _,dm_microns = AOsystem.set_dm_data(microns)
    image = Camera.take_image()
    image_bench = sf.reduce_images(image, xcen=xcen, ycen=ycen, npix=Npix_foc,
                                refpsf=OpticalModel.ref_psf.shaped,
                                )

    pupil_wf = hcipy.Wavefront(Aperture.aperture * np.exp(1j * add_mode),
                             wavelength=FnF.wavelength)
    focal_wf = OpticalModel.propagator(pupil_wf)

    image_theory = focal_wf.power

    # image_bench = sf.reduce_images(image, xcen=xcen, ycen=ycen, npix=Npix_foc,
    #                             refpsf=OpticalModel.ref_psf.shaped,
    #                             )
    
    plt.subplot(1, 2, 1)
    hcipy.imshow_field(np.log10(image_theory / image_theory.max()), vmin=-3)
    plt.colorbar()
    plt.title('theory')

    plt.subplot(1, 2, 2)
    plt.imshow(np.log10(np.abs(image_bench) / image_bench.max()), vmin=-3, origin='lower')
    plt.colorbar()
    plt.title('bench')
    plt.title('Applied ramdom zernikes command')

    plt.show()
    
    
    if aosystem == 'Sim':
        AOsystem.OpticalModel.update_pupil_wavefront(AOsystem.initial_phase_error)
    else:
        AOsystem.AO.revert_cog()
    
if __name__ == "__main__":
    run_fastandfurious_test(camera='Sim', 
                            aosystem='Sim', 
                            FF_ini = 'FF_software_sim.ini', 
                            FF_spec = 'FF_software.spec')