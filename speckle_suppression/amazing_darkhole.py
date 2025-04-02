
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
import ipdb
from datetime import datetime
import shutil 


def clamp(ref_psf, control_region, clamp=0):

    weight_in_control = ref_psf[control_region]
    weight_in_control[weight_in_control < clamp] = 0
    weight_in_control[weight_in_control >= clamp] = 1
    return weight_in_control

def condition(array, minimum=0, maximum=np.inf, minrep=0, maxrep=np.inf):
    """set any values less than minimum to minrep,
    and any values greater than maximum to maxrep"""
    array_copy = array.copy()
    array_copy[array_copy < minimum] = minrep
    array_copy[array_copy >= maximum] = maxrep
    return array_copy 

def condition_coeffs(coeffs):
    """conditions the sine and cosine coefficients"""
    coeffs_copy = coeffs.copy()
    coeffs_copy[np.isnan(coeffs_copy)] = 0
    coeffs_copy[np.isinf(coeffs_copy)] = 0
    return coeffs_copy

def amplitude_weight(ref_psf, control_region):
    """
    """
    # weight by electric field magnitude
    weight_in_control = ref_psf[control_region]
    weight_in_control = np.sqrt(np.abs(weight_in_control))
    max_in_control = weight_in_control.max()
    weight_in_control /= max_in_control
    
    return weight_in_control


if __name__ == "__main__":
    
    # make timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"output_{timestamp}"
    os.makedirs(dir_name, exist_ok=False)
    os.makedirs(os.path.join(dir_name,"bgds", exist_ok=False)
    
    config = 'sn_config.ini'
    configspec = 'sn_config.spec'
    Camera = hw.NIRC2Alias()
    AOSystem = hw.AOSystemAlias()
        
    settings = sf.validate_config(config, configspec)
    bgds = sf.setup_bgd_dict(settings['CAMERA_CALIBRATION']['bgddir'])
    
    IWA = settings['SN_SETTINGS']['IWA']
    OWA = settings['SN_SETTINGS']['OWA']
    
    # Save the configuration file used
    src = config
    src_destination = os.path.join(dir_name, src)
    shutil.copy(src, src_destination)
    
    src = configspec
    src_destination = os.path.join(dir_name, src)
    shutil.copy(src, src_destination)

    src = "amazing_darkhole.py"
    src_destination = os.path.join(dir_name, src)
    shutil.copy(src, src_destination)

    # Save the Backgrounds
    # src = settings['CAMERA_CALIBRATION']['bgddir']
    # dst = dir_name
    # shutil.copytree(src, os.path.join(dst, "bgds"))

    #----------------------------------------------------------------------
    # Simulation parameters
    #----------------------------------------------------------------------
    #CAMERA, AO, CORONAGRAPH SETTINGS IN CONFIG FILE
    #SN Settings
    xcen                = settings['SN_SETTINGS']['xcen']
    ycen                = settings['SN_SETTINGS']['ycen']
    cropsize            = settings['SN_SETTINGS']['cropsize']
    
    #DM Registration
    xcen                = settings['DM_REGISTRATION']['MEASURED_PARAMS']['centerx']
    ycen                = settings['DM_REGISTRATION']['MEASURED_PARAMS']['centery']
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
                    theta1=-75,
                    theta2=75,
    )
    
    hdu = fits.PrimaryHDU(control_region*1)
    hdu.writeto(os.path.join(dir_name,"controlregion.fits"), overwrite=True)
    print("WROTE CONTROL REGION") 
    #plt.figure()
    #plt.title(f"Median Counts in Control Region = {np.median(ref_img[control_region==1])}")
    #plt.imshow(ref_img * control_region, cmap="inferno", origin="lower", norm=LogNorm())
    #plt.colorbar()
    #ax = plt.gca()

    ## build the dark hole region
    #dh_region = Wedge([xcen, ycen], r=OWA_pix, width=OWA_pix-IWA_pix,
    #                  theta1=-90, theta2=90, facecolor="None", edgecolor="w")
    #ax.add_patch(dh_region)
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
    MAX_ITERS = 15
    
    hdu = fits.PrimaryHDU(ref_img)
    hdu.writeto(os.path.join(dir_name,"ref_img_Halfdark_ND1_5ms.fits"), overwrite=True)
    
    clamp_val = sf.robust_sigma(ref_img[:50, :50].ravel())
    probe_scaling_param = 1 
    plt.ion()
    fig, ax = plt.subplots()
    pixrad, clevel = sn_f.contrastcurve_simple(ref_img, 
                                               cx=xcen,
                                               cy=ycen,
                                               region=control_region*1,
                                               maxrad=100)
    line, = ax.plot(pixrad, clevel, label = 'Initial Contrast')
    plt.axhline(clamp_val, label = 'Bgd limit')
    plt.xlabel('pixels')
    plt.ylabel('1-sigma contrast')
    ax.legend()
    plt.draw()
    plt.pause(0.1)
    
    hdu = fits.PrimaryHDU(current_dm_shape)
    hdu.writeto(os.path.join(dir_name,"starting_dm_shape.fits"), overwrite=True)
    contrast_curves = []

    for i in range(MAX_ITERS):
        
        if i == 0:
            updated_dm_shape = current_dm_shape
        
        if i != 0:
            ref_img = sf.equalize_image(Camera.take_image(), **bgds)
        clamp_mask = clamp(ref_img, control_region, clamp=clamp_val)
        print(f"Pixels Clamped = {np.sum(1-clamp_mask)}")
        amp_mask = amplitude_weight(ref_img, control_region)

        # cosine probe
        cos_probe = cos_probe*probe_scaling_param
        set_shape = AOSystem.set_dm_data(updated_dm_shape + cos_probe)
        cos_plus_img = sf.equalize_image(Camera.take_image(), **bgds)
        
        # sine probe
        sin_probe = sin_probe*probe_scaling_param
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
        regularization = 0
        
        dE1 = (Ip1 - Im1) / 4
        dE2 = (Ip2 - Im2) / 4
        
        dE1sq = (Ip1 + Im1 - 2*I0) / 2
        dE2sq = (Ip2 + Im2 - 2*I0) / 2

        # # Regularized sin / cosine coefficients
        sin_coeffs = dE1 / dE1sq 
        sin_coeffs = condition_coeffs(sin_coeffs)
        cos_coeffs = dE2 / dE2sq 
        cos_coeffs = condition_coeffs(cos_coeffs)
        
        # plot the coefficients
        sin_coeffs_init = sin_coeffs
        cos_coeffs_init = cos_coeffs
        
        sin_coeffs_control = sin_coeffs[control_region] * clamp_mask * amp_mask
        cos_coeffs_control = cos_coeffs[control_region] * clamp_mask * amp_mask
        sin_coeffs_control = sin_coeffs_control[..., None, None]
        cos_coeffs_control = cos_coeffs_control[..., None, None]

        # create control modes

        sin_mode_control = np.sum(sin_coeffs_control * sine_modes, axis=0)
        cos_mode_control = np.sum(cos_coeffs_control * cosine_modes, axis=0)
        
        VOLT_THRESHOLD = 7
        MAX_CORRECTION = 0.1
        control_surface = -1 * (sin_mode_control + cos_mode_control)
        control_surface -= np.mean(control_surface)
        control_surface *= MAX_CORRECTION / np.max(np.abs(control_surface))*np.min([probe_scaling_param, 1])

        # Safety, threshold command greater than 7 volts 
        
        # Apply correction and take image
        updated_dm_shape = updated_dm_shape + control_surface
        set_shape = AOSystem.set_dm_data(updated_dm_shape)
        corrected_img = sf.equalize_image(Camera.take_image(), **bgds)
        
        print(20*"-")
        prev_sum = np.sum(ref_img[control_region])
        current_sum = np.sum(corrected_img[control_region])
        print(f"Original Sum in Region = {prev_sum}")
        print(f"Iteration {i} Sum in Region = {current_sum}")
        print(20*"-")
        
        pixrad, clevel = sn_f.contrastcurve_simple(ref_img, 
                                                   cx=xcen,
                                                   cy=ycen,
                                                   region=control_region*1,
                                                   maxrad=100)
        line, = ax.plot(pixrad, clevel, label = f'Iteration: {i}', alpha =0.5)
        plt.legend()
        plt.draw()
        plt.pause(0.1)
        contrast_curves.append(clevel)

        probe_scaling_param = np.sqrt(current_sum)/np.sqrt(prev_sum)
        print(f"Probe scaling param: {probe_scaling_param}")
        
        hdu = fits.PrimaryHDU(updated_dm_shape)
        hdu.writeto(os.path.join(dir_name,"SAN_iter{i}_dm_shape.fits"), overwrite=True)
        
        # write the images
        hdu = fits.PrimaryHDU(corrected_img)
        hdu.writeto(os.path.join(dir_name,f"SAN_iter{i}_Halfdark_ND1_5ms.fits"))
    
    plt.ioff()
    plt.close(fig)

    # Save Contrast Curves
    contrast_curves = np.asarrray(contrast_curves)
    hdu = fits.PrimaryHDU(contrast_curves)
    hdu.writeto(os.path.join(dir_name,"contrast_curves.fits"), overwrite=True)

    # FINAL: Return to flat
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

