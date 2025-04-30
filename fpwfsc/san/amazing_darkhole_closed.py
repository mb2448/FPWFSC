
############################### Import Library ################################

## Math Library
import numpy as np
from scipy.signal import convolve2d

## System library
import sys

## Operating system library
import os

## .fits file library
import astropy.io.fits as fits

## Other standards
import matplotlib.pyplot as plt 
from matplotlib.patches import Wedge
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ipdb
from time import sleep
from datetime import datetime
import shutil 

## FPWFSC Imports
# sys.path.insert(0,"../common")
from fpwfsc.common import support_functions as sf
from fpwfsc.common import bench_hardware as hw
from fpwfsc.san import sn_functions as sn_f
from fpwfsc.san import sn_classes as sn_c
from fpwfsc.common import dm
from fpwfsc.san.activation import neighbor_mask, arctan, tanh

def clamp(ref_psf, control_region, clamp=0):
    """
    Produces a binary amplitude mask where 1's are values above the specified
    clamp and 0's are values below the specified clamp

    Parameters
    ----------
    ref_psf : ndarray
        image to apply clamp to
    control_region : ndarray
        boolean array that contains the region of interest
    clamp: float
        value below which the mask returns a zero

    Returns
    -------
    ndarray
        1D array of binary weights of shape == where control region is 1
        
    """
    weight_in_control = ref_psf[control_region]
    weight_in_control[weight_in_control < clamp] = 0
    weight_in_control[weight_in_control >= clamp] = 1
    return weight_in_control

def condition(array, minimum=0, maximum=np.inf, minrep=0, maxrep=np.inf):
    """set any values less than minimum to minrep,
    and any values greater than maximum to maxrep
    
    Parameters
    ----------
    array : ndarray
        array to apply conditioning to
    minimum : float
        value below which the array is set to minrep
    maximum : float
        value above which the array is set to maxrep
    minrep : float
        value to replace elements in array that are below minimum
    maxrep : float
        value to replace elements in array that are above maximum
    
    Returns
    -------
    ndarray
        conditioned array subject to minimum and maximum
    """
    array_copy = array.copy()
    array_copy[array_copy < minimum] = minrep
    array_copy[array_copy >= maximum] = maxrep
    return array_copy 

def condition_coeffs(coeffs):
    """sets NaN sine and cosine coefficients to be zero"""
    coeffs_copy = coeffs.copy()
    coeffs_copy[np.isnan(coeffs_copy)] = 0
    coeffs_copy[np.isinf(coeffs_copy)] = 0
    return coeffs_copy

def amplitude_weight(ref_psf, control_region):
    """
    Creates array of amplitude weights where the maximum value is 1,
    and the remaining values are scaled by 1 / maximum value

    Parameters
    ----------
    ref_psf : ndarray
        image to apply clamp to
    control_region : ndarray
        boolean array that contains the region of interest
    
    Returns
    -------
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
    os.makedirs(os.path.join(dir_name,"bgds"), exist_ok=False)
    folder = 'fpwfsc/san/'
    config = 'sn_config.ini'
    configspec = 'sn_config.spec'
    Camera = hw.NIRC2Alias()
    #AOSystem = hw.AOSystemAlias()

    settings = sf.validate_config(folder+config, folder+configspec)
    bgds = sf.setup_bgd_dict(settings['CAMERA_CALIBRATION']['bgddir'])
    
    AOSystem = hw.ClosedAOSystemAlias()

    IWA = settings['SN_SETTINGS']['IWA']
    OWA = settings['SN_SETTINGS']['OWA']
    DH_THETA1 = settings['SN_SETTINGS']['THETA1']
    DH_THETA2 = settings['SN_SETTINGS']['THETA2'] 
    amplitude = settings['SN_SETTINGS']['DM_AMPLITUDE_VOLTS'] # volts
    
    # Save the configuration file used
    src = config
    src_destination = os.path.join(dir_name, src)
    shutil.copy(folder+src, src_destination)
    
    src = configspec
    src_destination = os.path.join(dir_name, src)
    shutil.copy(folder+src, src_destination)

    src = "amazing_darkhole.py"
    src_destination = os.path.join(dir_name, src)
    shutil.copy(folder+src, src_destination)

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
                    theta1=DH_THETA1,
                    theta2=DH_THETA2,
    )

    # construct a dark hole on the other side
    if settings['SN_SETTINGS']['FULL_DARKHOLE']:
        anti_control_region = sn_f.create_annular_wedge(
                        image=ref_img,
                        xcen=xcen, # pixels
                        ycen=ycen, # pixels
                        rad1=IWA_pix, # pixels
                        rad2=OWA_pix, # pixels
                        theta1=DH_THETA1 + 180,
                        theta2=DH_THETA2 + 180,
        )

        full_control_region = control_region + anti_control_region
    
    else:
        full_control_region = control_region

    hdu = fits.PrimaryHDU(full_control_region*1)
    hdu.writeto(os.path.join(dir_name,"controlregion.fits"), overwrite=True)
    print("WROTE CONTROL REGION") 

    ## Build the probes

    control_indices = np.where(control_region)
    sine_modes = []
    cosine_modes = []
    
    control_pix_x = []
    control_pix_y = []
    
    anti_control_pix_x = []
    anti_control_pix_y = []

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

    MAX_ITERS = settings['SN_SETTINGS']['NUM_ITERATIONS']

    hdu = fits.PrimaryHDU(ref_img)
    hdu.writeto(os.path.join(dir_name,"ref_img_Halfdark_ND1_5ms.fits"), overwrite=True)
    
    clamp_val = sf.robust_sigma(ref_img[:50, :50].ravel())
    probe_scaling_param = 1 
    plt.ion()
    fig, ax = plt.subplots(figsize=[10, 5], ncols=2)
    pixrad, clevel = sn_f.contrastcurve_simple(ref_img, 
                                               cx=xcen,
                                               cy=ycen,
                                               region=full_control_region*1,
                                               maxrad=OWA * lambdaoverd + 10)
    line, = ax[0].plot(pixrad, clevel, label = 'Initial Contrast')
    ax[0].axhline(clamp_val, label = 'Bgd limit')
    ax[0].set_xlabel('pixels')
    ax[0].set_ylabel('1-sigma contrast')
    ax[0].legend()
    
    hdu = fits.PrimaryHDU(current_dm_shape)
    hdu.writeto(os.path.join(dir_name,"starting_dm_shape.fits"), overwrite=True)
    contrast_curves = []

    if AOSystem._closed:
        cos_probe = AOSystem.convert_voltage_to_cog(cos_probe)
        sin_probe = AOSystem.convert_voltage_to_cog(sin_probe)
    
    for i in range(MAX_ITERS):
        
        if i == 0:
            updated_dm_shape = current_dm_shape
        
        if i != 0:
            ref_img = sf.equalize_image(Camera.take_image(), **bgds)

        vmin, vmax = np.percentile(ref_img, [0, 100])
        linthresh = 0.01 * max(np.abs(vmin), np.abs(vmax))
        norm = SymLogNorm(vmin=vmin, vmax=vmax, linthresh=linthresh)
        im = ax[1].imshow(ref_img, norm=norm)
        CROP_RAD = 64
        ax[1].set_xlim(xcen-CROP_RAD, xcen+CROP_RAD)
        ax[1].set_ylim(ycen-CROP_RAD, ycen+CROP_RAD)
        div = make_axes_locatable(ax[1])
        cax = div.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        th1 = settings["SN_SETTINGS"]["THETA1"]
        th2 = settings["SN_SETTINGS"]["THETA2"]
        dh_region = Wedge([xcen, ycen], r=OWA_pix, width=OWA_pix-IWA_pix,
                        theta1=th1, theta2=th2, facecolor="None", edgecolor="w")
        ax[1].add_patch(dh_region)
        plt.draw()
        plt.pause(0.1)

        clamp_mask = clamp(ref_img, control_region, clamp=clamp_val)
        print(f"Pixels Clamped = {np.sum(1-clamp_mask)}")
        amp_mask = 1 # amplitude_weight(ref_img, control_region)

        # cosine probe
        cos_probe = cos_probe*probe_scaling_param
        sin_probe = sin_probe*probe_scaling_param


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
        
        if settings['SN_SETTINGS']['FULL_DARKHOLE']:

            ref_img_rotated = sn_f.flip_array_about_point(ref_img, xcen, ycen)
            cos_minus_img_rotated = sn_f.flip_array_about_point(cos_minus_img, xcen, ycen)
            sin_minus_img_rotated = sn_f.flip_array_about_point(sin_minus_img, xcen, ycen)
            sin_plus_img_rotated = sn_f.flip_array_about_point(sin_plus_img, xcen, ycen)
            cos_plus_img_rotated = sn_f.flip_array_about_point(cos_plus_img, xcen, ycen)
            
            # Compute the relevant quantities
            # 1- sin quantities, 2- cosine quantities
            Ip1 = (sin_plus_img + sin_plus_img_rotated) / 2
            Im1 = (sin_minus_img + sin_minus_img_rotated) / 2
            Ip2 = (cos_plus_img + cos_plus_img_rotated) / 2
            Im2 = (cos_minus_img + cos_minus_img_rotated) / 2
            I0 = (ref_img + ref_img_rotated) / 2
        
        else:

            # Compute the relevant quantities
            # 1- sin quantities, 2- cosine quantities
            Ip1 = sin_plus_img 
            Im1 = sin_minus_img
            Ip2 = cos_plus_img
            Im2 = cos_minus_img
            I0 = ref_img


        dE1 = (Ip1 - Im1) / 4
        dE2 = (Ip2 - Im2) / 4
        
        dE1sq = (Ip1 + Im1 - 2*I0) / 2
        dE2sq = (Ip2 + Im2 - 2*I0) / 2

        # # Regularized sin / cosine coefficients
        sin_coeffs = dE1 / dE1sq 
        sin_coeffs = condition_coeffs(sin_coeffs)
        cos_coeffs = dE2 / dE2sq 
        cos_coeffs = condition_coeffs(cos_coeffs)
        
        # coeffs that get plotted
        sin_coeffs_init = sin_coeffs
        cos_coeffs_init = cos_coeffs

        neighbors_weight = neighbor_mask(control_region)
        sin_coeffs *= neighbors_weight
        cos_coeffs *= neighbors_weight 
        sin_coeffs_control = sin_coeffs[control_region] * clamp_mask * amp_mask
        cos_coeffs_control = cos_coeffs[control_region] * clamp_mask * amp_mask
        sin_coeffs_control = sin_coeffs_control[..., None, None]
        cos_coeffs_control = cos_coeffs_control[..., None, None]

        # create control modes
        sin_mode_control = np.sum(sin_coeffs_control * sine_modes, axis=0)
        cos_mode_control = np.sum(cos_coeffs_control * cosine_modes, axis=0)
        
        # NOTE algorithm is sensitive to this parameter
        # in closed loop, 0.3 was good when not using the activation
        # arctan : hot speckles appeared - perhaps because of quadrant?
        # tanh : 
        #    - 0.1 is very slow 
        #    - 0.3 did not really converge 
        MAX_CORRECTION = 0.2 # was 0.1, for slow convergence
        
        control_surface = -1 * (sin_mode_control + cos_mode_control)
        control_surface -= np.mean(control_surface)
        # control_surface *= MAX_CORRECTION / np.max(np.abs(control_surface))*np.min([probe_scaling_param, 1])
        control_surface = MAX_CORRECTION * tanh(control_surface, a=MAX_CORRECTION) * \
                          np.min([probe_scaling_param, 1]) / np.pi * 2
        # Safety, threshold command greater than 7 volts 
        
        # Apply correction and take image
        if AOSystem._closed:
            control_surface = AOSystem.convert_voltage_to_cog(control_surface)

        updated_dm_shape = updated_dm_shape + control_surface
        set_shape = AOSystem.set_dm_data(updated_dm_shape)
        corrected_img = sf.equalize_image(Camera.take_image(), **bgds)
        
        print(20*"-")
        prev_sum = np.sum(ref_img[control_region])
        current_sum = np.sum(corrected_img[control_region])
        print(f"Original Sum in Region = {prev_sum}")
        print(f"Iteration {i} Sum in Region = {current_sum}")
        print(20*"-")
        
        pixrad, clevel = sn_f.contrastcurve_simple(corrected_img, 
                                                   cx=xcen,
                                                   cy=ycen,
                                                   region=full_control_region*1,
                                                   maxrad=OWA * lambdaoverd + 10)
        
        line, = ax[0].plot(pixrad, clevel, label = f'Iteration: {i}', alpha =0.5)
        ax[0].legend()
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

        intermediates = [I0, Im1, Ip1, Im2, Ip2]
        intermediates = np.asarray(intermediates)
        hdu = fits.PrimaryHDU(intermediates)
        hdu.header["IM1"] = "I0"
        hdu.header["IM2"] = "Im1 Sin"
        hdu.header["IM3"] = "Ip1 Sin"
        hdu.header["IM4"] = "Im2 Cos"
        hdu.header["IM5"] = "Ip2 Cos"
        hdu.writeto(os.path.join(dir_name,f"SAN_iter{i}_intermediates_ND1_5ms.fits"))
            
    # plt.ioff()
    # plt.close(fig)

    # Save Contrast Curves
    contrast_curves.append(np.ones_like(contrast_curves[0]) * clamp_val)
    contrast_curves = np.asarray(contrast_curves)
    hdu = fits.PrimaryHDU(contrast_curves)
    hdu.writeto(os.path.join(dir_name,"contrast_curves.fits"), overwrite=True)

    # FINAL: Return to flat
    set_shape = AOSystem.set_dm_data(current_dm_shape) 

    #FINAL: Plot all figures
    if not AOSystem._closed: 
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

