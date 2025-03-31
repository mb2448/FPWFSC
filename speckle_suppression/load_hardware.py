
############################### Import Library ################################

## Math Library
import numpy as np
## System library
import sys
## Operating system library
import os
## .fits file library
import astropy.io.fits as fits
import matplotlib.pyplot as plt 
from matplotlib.patches import Wedge
from matplotlib.colors import LogNorm
sys.path.insert(0,"../common")
import support_functions as sf
import bench_hardware as hw
import sn_functions as sn_f
import sn_classes as sn_c
import dm
from time import sleep

if __name__ == "__main__":
    IWA = 4
    OWA = 7
    config = 'sn_config.ini'
    configspec = 'sn_config.spec'
    Camera = hw.NIRC2Alias()
    AOSystem = hw.AOSystemAlias()
        
    settings = sf.validate_config(config, configspec)
    bgds = sf.setup_bgd_dict(settings['CAMERA_CALIBRATION']['bgddir'])
    #----------------------------------------------------------------------
    # Simulation parameters
    #----------------------------------------------------------------------
    #CAMERA, AO, CORONAGRAPH SETTINGS IN CONFIG FILE
    #SN Settings
    xcen                = settings['SN_SETTINGS']['xcen']
    ycen                = settings['SN_SETTINGS']['ycen']
    cropsize            = settings['SN_SETTINGS']['cropsize']
    
    #DM Registration
    xcen                = settings['DM_REGISTRATION']['MEASURED_PARAMS']['center_x']
    ycen                = settings['DM_REGISTRATION']['MEASURED_PARAMS']['center_y']
    dm_angle            = settings['DM_REGISTRATION']['MEASURED_PARAMS']['angle']
    lambdaoverd         = settings['DM_REGISTRATION']['MEASURED_PARAMS']['lambdaoverd']

    # Draw a dark hole

    # convert IWA/OWA to pixels
    IWA_pix = IWA * lambdaoverd
    OWA_pix = OWA * lambdaoverd 
    ref_img = sf.equalize_image(Camera.take_image(), **bgds) 
    control_region = sn_f.create_annular_wedge(
                    image=ref_img,
                    xcen=xcen, # pixels
                    ycen=ycen, # pixels
                    rad1=IWA_pix, # pixels
                    rad2=OWA_pix, # pixels
                    theta1=-90,
                    theta2=90
    )

    plt.figure()
    plt.imshow(ref_img, cmap="inferno", origin="lower", norm=LogNorm())
    plt.colorbar()
    ax = plt.gca()

    # build the dark hole region
    dh_region = Wedge([xcen, ycen], r=OWA_pix, width=OWA_pix-IWA_pix,
                      theta1=-90, theta2=90, facecolor="None", edgecolor="w")
    ax.add_patch(dh_region)
    # plt.show()

    ## Build the probes
    amplitude = 1 # volts

    #SAN = sn_c.SpeckleAreaNulling(
    #        camera=Camera,
    #        aosystem=AOSystem,
    #        initial_probe_amplitude=amplitude,
    #        controlregion_iwa=IWA,
    #        controlregion_owa=OWA,
    #        xcenter=xcen,
    #        ycenter=ycen,
    #        Npix_foc=cropsize,
    #        lambdaoverD=lambdaoverd)
    

    control_indices = np.where(control_region)
    sine_modes = []
    cosine_modes = []

    for yi, xi in zip(*control_indices):

        # construct probes from the wedge
        cos = dm.make_speckle_xy(
                xs=xi,
                ys=yi,
                amps=1,
                phases=0,
                centerx=xcen,
                centery=ycen,
                angle=dm_angle,
                lambdaoverd=lambdaoverd,
                N=21,
                which="cos")
        
        cosine_modes.append(cos)

        sin = dm.make_speckle_xy(
                xs=xi,
                ys=yi,
                amps=1,
                phases=0,
                centerx=xcen,
                centery=ycen,
                angle=dm_angle,
                lambdaoverd=lambdaoverd,
                N=21,
                which="sin")
        
        sine_modes.append(sin)
    
    
    # Construct and scale the probes from the modes
    cos_probe = np.sum(cosine_modes, axis=0)
    sin_probe = np.sum(sine_modes, axis=0)

    cos_probe *= amplitude / cos_probe.max()
    sin_probe *= amplitude / sin_probe.max()
    
    cosine_modes = np.asarray(cosine_modes) * amplitude / cos_probe.max()
    sine_modes = np.asarray(sine_modes) * amplitude / sin_probe.max()

    # Take the probe measurements
    # NOTE: ref_img is the un-probed psf

    # apply DM shape
    current_dm_shape = AOSystem.get_dm_data()
    MAX_ITERS = 5

    for i in range(MAX_ITERS):
        
        if i == 0:
            updated_dm_shape = current_dm_shape

        # cosine probe
        set_shape = AOSystem.set_dm_data(updated_dm_shape + cos_probe)
        cos_plus_img = sf.equalize_image(Camera.take_image(), **bgds)
        
        # sine probe
        set_shape = AOSystem.set_dm_data(updated_dm_shape + sin_probe)
        sin_plus_img = sf.equalize_image(Camera.take_image(), **bgds)
        
        # cosine probe
        set_shape = AOSystem.set_dm_data(updated_dm_shape - cos_probe)
        cos_minus_img = sf.equalize_image(Camera.take_image(), **bgds)
        
        # sine probe
        set_shape = AOSystem.set_dm_data(updated_dm_shape - sin_probe)
        sin_minus_img = sf.equalize_image(Camera.take_image(), **bgds)

        # Compute the relevant quantities
        # 1- sin quantities, 2- cosine quantities
        Ip1 = sin_plus_img
        Im1 = sin_minus_img
        Ip2 = cos_plus_img
        Im2 = cos_minus_img
        I0 = ref_img
        regularization = 100

        dE1 = (Ip1 - Im1) / 4
        dE2 = (Ip2 - Im2) / 4
        dE1sq = (Ip1 + Im1 - 2*I0) / 2
        dE2sq = (Ip2 + Im2 - 2*I0) / 2

        # # Regularized sin / cosine coefficients
        sin_coeffs = dE1 / (dE1sq + regularization)
        cos_coeffs = dE2 / (dE2sq + regularization)
        
        # plot the coefficients
        sin_coeffs_init = sin_coeffs
        cos_coeffs_init = cos_coeffs

        sin_coeffs_control = sin_coeffs[control_region]
        cos_coeffs_control = cos_coeffs[control_region]
        sin_coeffs_control = sin_coeffs_control[..., None, None]
        cos_coeffs_control = cos_coeffs_control[..., None, None]

        # create control modes
        sin_mode_control = np.sum(sin_coeffs_control * sine_modes, axis=0)
        cos_mode_control = np.sum(cos_coeffs_control * cosine_modes, axis=0)
        
        VOLT_THRESHOLD = 7
        MAX_CORRECTION = 0.1
        control_surface = -1 * (sin_mode_control + cos_mode_control)
        control_surface -= np.mean(control_surface)
        control_surface *= MAX_CORRECTION / np.max(np.abs(control_surface))

        # Safety, threshold command greater than 7 volts 
        
        # Apply correction and take image
        updated_dm_shape = updated_dm_shape + control_surface
        set_shape = AOSystem.set_dm_data(updated_dm_shape)
        corrected_img = sf.equalize_image(Camera.take_image(), **bgds)
        print(20*"-")
        print(f"ITERATION {i+1} DONE WOO")
        print(20*"-")
    #FINAL: Return to flat
    set_shape = AOSystem.set_dm_data(current_dm_shape) 
    
    #FINAL: Plot all figures
    
    plt.figure(figsize=[10, 5])
    plt.subplot(121)
    plt.title("Differential sin image")
    plt.imshow(sin_plus_img - ref_img, cmap="coolwarm", vmin=-200, vmax=200)
    plt.colorbar()
    ax = plt.gca()
    # build the dark hole region
    dh_region = Wedge([xcen, ycen], r=OWA_pix, width=OWA_pix-IWA_pix,
                      theta1=-90, theta2=90, facecolor="None", edgecolor="w")
    ax.add_patch(dh_region)
    
    plt.subplot(122)
    plt.title("Differential cosine image")
    plt.imshow(cos_plus_img - ref_img, cmap="coolwarm", vmin=-200, vmax=200)
    plt.colorbar()
    ax = plt.gca()
    # build the dark hole region
    dh_region = Wedge([xcen, ycen], r=OWA_pix, width=OWA_pix-IWA_pix,
                      theta1=-90, theta2=90, facecolor="None", edgecolor="w")
    ax.add_patch(dh_region)
    plt.figure(figsize=[10, 5])
    plt.subplot(121)
    plt.title("SAN Control Surface")
    plt.imshow(control_surface, cmap="RdBu_r")
    plt.colorbar(label="Volts")
    
    plt.subplot(122)
    plt.title("Image Correction")
    plt.imshow(corrected_img, cmap="inferno", norm=LogNorm())
    plt.colorbar()
    ax = plt.gca()
    # build the dark hole region
    dh_region = Wedge([xcen, ycen], r=OWA_pix, width=OWA_pix-IWA_pix,
                      theta1=-90, theta2=90, facecolor="None", edgecolor="w")
    ax.add_patch(dh_region)
    
    pushpulls = [Ip1, Im1, Ip2, Im2]
    plt.figure(figsize=[7, 7])
    for i, img in enumerate(pushpulls):
        plt.subplot(2, 2, i+1)
        plt.imshow(img, cmap="inferno", norm=LogNorm())
        plt.colorbar()
        ax = plt.gca()
        dh_region = Wedge([xcen, ycen], r=OWA_pix, width=OWA_pix-IWA_pix,
                          theta1=-90, theta2=90, facecolor="None", edgecolor="w")
        ax.add_patch(dh_region)
    

    plt.figure(figsize=[10, 5])
    plt.subplot(121)
    plt.title("SAN sin coefficients")
    plt.imshow(sin_coeffs_init, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    ax = plt.gca()
    # build the dark hole region
    dh_region = Wedge([xcen, ycen], r=OWA_pix, width=OWA_pix-IWA_pix,
                      theta1=-90, theta2=90, facecolor="None", edgecolor="w")
    ax.add_patch(dh_region)
    
    plt.subplot(122)
    plt.title("SAN cos coefficients")
    plt.imshow(cos_coeffs_init, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    ax = plt.gca()
    # build the dark hole region
    dh_region = Wedge([xcen, ycen], r=OWA_pix, width=OWA_pix-IWA_pix,
                      theta1=-90, theta2=90, facecolor="None", edgecolor="w")
    ax.add_patch(dh_region)
    plt.show()

