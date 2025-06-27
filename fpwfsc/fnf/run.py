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
import json
import os

from astropy.io import fits

from ..common import plotting_funcs as pf
from ..common import classes as ff_c
from ..common import fake_hardware as fhw
from ..common import support_functions as sf
from ..san import sn_functions as sn

def run(camera=None, aosystem=None, config=None, configspec=None,
        my_deque=None, my_event=None, plotter=None, ):
    if my_deque is None:
        my_deque = deque()

    if my_event is None:
        my_event = threading.Event()
    settings = sf.validate_config(config, configspec)
    print('*************************************************************************')
    print(type(config))
    

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
                                    initial_rms_wfe=rms_wfe, seed=seed)
                                    #rotation_angle_dm = rotation_angle_dm)
    else:
        Camera = camera
        AOsystem = aosystem
    # generating the first reference image




    bgds = sf.setup_bgd_dict('../bgds/')
    #print(bgds)
    #XXX! Add this to the hardware file, self.get_nest_filename = self.nirc2.get_next_flename
    #filename = Camera.get_next_filename()

    data_raw = Camera.take_image()
    if data_raw.ndim != 2:
        data_raw = data_raw[0]
    data_ref = sf.reduce_images(data_raw, xcen=xcen, ycen=ycen,
                                          npix=Npix_foc,
                                          refpsf=OpticalModel.ref_psf.shaped,
                                          bgds = bgds
                                          )
    # Take first image
    FnF.initialize_first_image(data_ref)
    #MAIN LOOP SETUP AND RUNNING
    #----------------------------------------------------------------------

    SRA_measurements = np.zeros(Niter)
    VAR_measurements = np.zeros(Niter)

    SRA_measurements[SRA_measurements==0] = np.nan
    VAR_measurements[VAR_measurements==0] = np.nan
    t0 = time.time()



    #initialization for optional things 
    hitchhiker_mode = False
    if hitchhiker_mode==True:
        Hitch = fhw.Hitchhiker(imagedir='Hitchhiker_img')

    save_log = True
    if save_log == True: 
        logger = LogManager(base_log_dir="run_20250625_logs", config=settings)


    for i in np.arange(Niter):
        #The next two lines stop it if the user presses stop in the gui
        if my_event.is_set():
            return

        SRA_measurements[i] = FnF.estimate_strehl()
        my_deque.append(SRA_measurements)

        if hitchhiker_mode==True:
            #XXX for hitchhiker, we need to output the file name, shoild be in _read_fits, but other places are using the hitchiker so...
            img= Hitch.wait_for_next_image()
        else:
            #filename = Camera.get_next_filename()
            img = Camera.take_image()

        if img.ndim != 2:
            img = img[0]

        data = sf.reduce_images(img, xcen=xcen, ycen=ycen, npix=Npix_foc,
                                refpsf=OpticalModel.ref_psf.shaped,
                                bgds = bgds
                                )
        #update the loop with the new data
        phase_DM = FnF.iterate(data)
        #convert to usable DM units
        microns = -3 * phase_DM * FnF.wavelength / (2 * np.pi) * 1e6

        AO_cog, _ = AOsystem.set_dm_data(microns)




        # Saving metrics of strehl, airy ring variation
        VAR_measurements[i] = sf.calculate_VAR(data, OpticalModel.ref_psf.shaped,
                                               mas_pix, wavelength,
                                               Aperture.pupil_diameter)
        print('Strehl:', SRA_measurements[i], ";  VAR: ", VAR_measurements[i])

        pix_dis,contrast_measurements = sn.contrastcurve_simple(data, cx=None, cy = None, sigmalevel = 1, robust=True, region =None, maxrad = Npix_foc/2-1)
        mas_dis = pix_dis * mas_pix

        if plotter is not None:
            plotter.update(Niter=Niter,
                                data=FnF.previous_image,
                                pupil_wf=phase_DM,
                                aperture=FnF.aperture,
                                SRA=SRA_measurements,
                                separation = mas_dis,
                                contrast = contrast_measurements)

        logger.save_iteration(i, 
                       strehl=SRA_measurements[i],
                       contrast_curve=contrast_measurements,
                       separation=mas_dis,
                       ref_psf=OpticalModel.ref_psf.shaped,
                       phase_estimate=phase_DM.shaped,
                       dm_command=AO_cog.shaped, ##not sure what the real one looks like
                       raw_data=img,
                       processed_data=data,
                       raw_file=None,
                       backgrounds = bgds)
        

    t1 = time.time()
    print(str(Niter), ' iterations completed in: ', t1-t0, ' seconds')
    #AOsystem.close_dm_stream()???


class LogManager:
    def __init__(self, base_log_dir="logs", config=None):
        self.base_log_dir = base_log_dir
        os.makedirs(self.base_log_dir, exist_ok=True)
        
        # Save config once if provided
        if config:
            config_path = os.path.join(self.base_log_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

    def save_iteration(self, iter_num, *, 
                       strehl=None,
                       contrast_curve=None,
                       separation=None,
                       ref_psf=None,
                       phase_estimate=None,
                       dm_command=None,
                       raw_data=None,
                       processed_data=None,
                       raw_file=None,
                       backgrounds = None
                       ):

        iter_dir = os.path.join(self.base_log_dir, f"iter_{iter_num:03d}")
        os.makedirs(iter_dir, exist_ok=True)

        # Save metadata
        meta = {
            "iteration": int(iter_num),
            "strehl_ratio": float(strehl) if strehl is not None else None,
            "raw_file": raw_file,
            "backgrounds": backgrounds
            
        }
        with open(os.path.join(iter_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Save contrast curve (1D)
        if contrast_curve is not None and separation is not None:
            fits.writeto(os.path.join(iter_dir, "contrast.fits"),
                         np.array([separation, contrast_curve]),
                         overwrite=True)

        # Save reference PSF  
        if phase_estimate is not None:
            fits.writeto(os.path.join(iter_dir, "reference PSF.fits"),
                         ref_psf, overwrite=True)


        # Save 2D arrays
        if phase_estimate is not None:
            fits.writeto(os.path.join(iter_dir, "phase_estimate.fits"),
                         phase_estimate, overwrite=True)

        if dm_command is not None:
            fits.writeto(os.path.join(iter_dir, "dm_command.fits"),
                         dm_command, overwrite=True)

        # Save images only if data is provided
        if raw_data is not None:
            fits.writeto(os.path.join(iter_dir, "raw.fits"), raw_data, overwrite=True)

        if processed_data is not None:
            fits.writeto(os.path.join(iter_dir, "processed.fits"), processed_data, overwrite=True)
    

if __name__ == "__main__":
    plotter = pf.LivePlotter()
    camera = "Sim"
    aosystem = "Sim"

    script_dir = Path(__file__).parent
    config_path = script_dir/"FF_software_sim.ini"
    spec_path   = script_dir/"FF_software.spec"
    run(camera, aosystem, config=str(config_path),
                          configspec=str(spec_path))
