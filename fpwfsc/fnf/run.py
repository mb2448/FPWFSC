#!/usr/bin/env python
import sys
import threading
import numpy as np
from collections import deque
import hcipy
from configobj import ConfigObj
import time
import matplotlib.pyplot as plt
from pathlib import Path

from ..common import plotting_funcs as pf
from ..common import classes as ff_c
from ..common import fake_hardware as fhw
from ..common import support_functions as sf

def run(camera=None, aosystem=None, config=None, configspec=None,
        my_deque=None, my_event=None, plotter=None):
    if my_deque is None:
        my_deque = deque()

    if my_event is None:
        my_event = threading.Event()
    settings = sf.validate_config(config, configspec)

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
    rotation_primary_deg_pre  = 0
    rotation_primary_threshold = 5

    Aperture = ff_c.Aperture(Npix_pup=Npix_pup,
                             aperturename=chosen_aperture,
                             rotation_angle_aperture=rotation_angle_aperture,
                             rotation_primary_deg = rotation_primary_deg_pre)

    OpticalModel = ff_c.SystemModel(aperture=Aperture,
                                    Npix_foc=Npix_foc,
                                    mas_pix=mas_pix,
                                    wavelength=wavelength)
    
    Aperture2 = ff_c.Aperture(Npix_pup=Npix_pup,
                             aperturename='keck+OSIRIS_35_50_mas',
                             rotation_angle_aperture=60,
                             rotation_primary_deg = 50)

    OpticalModel2 = ff_c.SystemModel(aperture=Aperture2,
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
                                    initial_rms_wfe=rms_wfe, seed=seed)
                                    #rotation_angle_dm = rotation_angle_dm)
    else:
        Camera = camera
        AOsystem = aosystem
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

    SRA_measurements = np.zeros(Niter)
    VAR_measurements = np.zeros(Niter)

    SRA_measurements[SRA_measurements==0] = np.nan
    VAR_measurements[VAR_measurements==0] = np.nan
    t0 = time.time()

    test_rot = np.arange(100)


    for i in np.arange(Niter):
        #The next two lines stop it if the user presses stop in the gui
        if my_event.is_set():
            return

        SRA_measurements[i] = FnF.estimate_strehl()
        my_deque.append(SRA_measurements)


        #check if the change in primary mirror rotation has surpass the threshold required to regenerate the reference
        rotation_primary_deg_cur = test_rot[i]
        
        if np.abs(rotation_primary_deg_pre-rotation_primary_deg_cur) > rotation_primary_threshold:
            new_Aperture = ff_c.Aperture(Npix_pup=Npix_pup,
                             aperturename=chosen_aperture,
                             rotation_angle_aperture=rotation_angle_aperture,
                             rotation_primary_deg = rotation_primary_deg_cur)

            new_OpticalModel = ff_c.SystemModel(aperture=new_Aperture,
                                    Npix_foc=Npix_foc,
                                    mas_pix=mas_pix,
                                    wavelength=wavelength)
            
            rotation_primary_deg_pre = rotation_primary_deg_cur

            
            #Camera.opticalsystem=new_OpticalModel
            #AOsystem.OpticalModel = new_OpticalModel
            
            FnF.SystemModel = new_OpticalModel
            focalfield = new_OpticalModel.generate_psf_efield()
            focalimg = np.abs(focalfield.intensity)**2
            plt.figure()
            hcipy.imshow_field(np.log(focalimg))
            plt.colorbar()
            plt.title('focalimg '+str(i))
            plt.savefig(f'C:/UHManoa/First/focalimg{i}.png')

            


        img = Camera.take_image()
        data = sf.reduce_images(img, xcen=xcen, ycen=ycen, npix=Npix_foc,
                                flipx = flip_x, flipy=flip_y,
                                refpsf=OpticalModel.ref_psf.shaped,
                                rotation_angle_deg= rotation_angle_deg)
        #update the loop with the new data
        phase_DM = FnF.iterate(data)
        #convert to usable DM units
        microns = phase_DM * FnF.wavelength / (2 * np.pi) * 1e6
        dm_microns = AOsystem.make_dm_command(microns)
        AOsystem.set_dm_data(dm_microns)

        # Saving metrics of strehl, airy ring variation
        VAR_measurements[i] = sf.calculate_VAR(data, OpticalModel.ref_psf.shaped,
                                               mas_pix, wavelength,
                                               Aperture.pupil_diameter)
        print('Strehl:', SRA_measurements[i], ";  VAR: ", VAR_measurements[i])
        if plotter is not None:
            plotter.update(Niter=Niter,
                                data=FnF.previous_image,
                                pupil_wf=phase_DM,
                                aperture=FnF.aperture,
                                SRA=SRA_measurements,
                                VAR=VAR_measurements)
    t1 = time.time()
    print(str(Niter), ' iterations completed in: ', t1-t0, ' seconds')
    AOsystem.close_dm_stream()

if __name__ == "__main__":
    plotter = pf.LivePlotter()
    camera = "Sim"
    aosystem = "Sim"

    script_dir = Path(__file__).parent
    config_path = script_dir/"FF_software_sim.ini"
    spec_path   = script_dir/"FF_software.spec"
    run(camera, aosystem, config=str(config_path),
                          configspec=str(spec_path))
