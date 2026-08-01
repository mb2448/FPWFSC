
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
from fpwfsc.common import dm

# TODO: Update speckle nulling functions
from fpwfsc.san.activation import neighbor_mask, arctan, tanh
from fpwfsc.san import sn_functions as sn_f
from fpwfsc.san import sn_classes as sn_c

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


def condition_coeffs(coeffs, maxval):
    """sets NaN sine and cosine coefficients to be zero"""
    coeffs_copy = coeffs.copy()
    coeffs_copy[np.isnan(coeffs_copy)] = 0
    coeffs_copy[np.isinf(coeffs_copy)] = 0

    # clamp the coefficients to a maximum value
    coeffs_copy[np.abs(coeffs_copy) > maxval] = maxval \
        * np.sign(coeffs_copy[np.abs(coeffs_copy) > maxval])

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
    
    # User options are pulled from sn_config.ini
    # make timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = "sky_data/"
    work_dir = os.path.join(dir_name, timestamp)
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(os.path.join(dir_name,"bgds"), exist_ok=True)
    folder = '/home/mcisse/FPWFSC/fpwfsc/san/'
    config = 'sn_config.ini'
    configspec = 'sn_config.spec'
    Camera = hw.NIRC2Alias()

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
    src_destination = os.path.join(folder, work_dir)
    os.makedirs(src_destination, exist_ok=True)

    file_destination = os.path.join(src_destination, src)
    shutil.copy(src, file_destination)
    
    src = configspec
    file_destination = os.path.join(src_destination, src)
    shutil.copy(src, file_destination)

    src = "san_onsky_0.py"
    file_destination = os.path.join(src_destination, src)
    shutil.copy(src, file_destination)

    #CAMERA, AO, CORONAGRAPH SETTINGS IN CONFIG FILE
    #SN Settings

    # NOTE: These are overwrritten by the DM registration
    xcen                = settings['SN_SETTINGS']['xcen']
    ycen                = settings['SN_SETTINGS']['ycen']

    # Not overwritten
    cropsize            = settings['SN_SETTINGS']['cropsize']
    
    #DM Registration
    xcen                = settings['DM_REGISTRATION']['MEASURED_PARAMS']['centerx']
    ycen                = settings['DM_REGISTRATION']['MEASURED_PARAMS']['centery']
    dm_angle            = settings['DM_REGISTRATION']['MEASURED_PARAMS']['angle']
    lambdaoverd         = settings['DM_REGISTRATION']['MEASURED_PARAMS']['lambdaoverd']

    #DM Calibration
    parabola_params     = settings['DM_REGISTRATION']['INTENSITY_CAL']['parabola_parameters']

    # Draw a dark hole
    def take_image_save_qacits(dir_name, iter, timestamp, bgds):
        img = sf.equalize_image(Camera.take_image(), **bgds)
        
        # write the images
        hdu = fits.PrimaryHDU(img)
        hdu.writeto(os.path.join(dir_name,f"qacits_psfs/iter{iter}_{timestamp}.fits"))

        return img


    # convert IWA/OWA to pixels
    IWA_pix = IWA * lambdaoverd
    OWA_pix = OWA * lambdaoverd 
    #ref_img = sf.equalize_image(Camera.take_image(), **bgds)
    ref_img = take_image_save_qacits(dir_name, "ref", timestamp, bgds)

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
    hdu.writeto(os.path.join(work_dir,"controlregion.fits"), overwrite=True)

    ## Build the probes
    control_indices = np.where(control_region)
    sine_modes = []
    cosine_modes = []
    
    control_pix_x = []
    control_pix_y = []
    
    anti_control_pix_x = []
    anti_control_pix_y = []

    # Putting satelite spot on DM
    # print("Putting SATSPOT on DM")
    # specx = dm.make_speckle_kxy(
    #     kx=16,
    #     ky=0,
    #     amp=1,
    #     phase=0,
    #     N=60,
    #     centerx=xcen,
    #     centery=ycen,
    #     rotation=dm_angle,
    # )
    # 
    # specy = dm.make_speckle_kxy(
    #     kx=0,
    #     ky=16,
    #     amp=1,
    #     phase=0,
    #     N=60,
    #     centerx=xcen,
    #     centery=ycen,
    #     rotation=dm_angle,
    # )

    # satspots = specx + specy

    for yi, xi in zip(*control_indices):
        
        focal_amplitude = 1 #ref_img[yi, xi]

        # construct the amplitude weight
        kx, ky = dm.convert_pixels_kvecs(
                xi, yi,
                centerx=xcen,
                centery=ycen,
                angle=dm_angle,
                lambdaoverd=lambdaoverd)
        
        kr = np.sqrt(kx**2 + ky**2)
        speckle_amp = 1 #sn_f.parabola(kr, *parabola_params)

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
                N=60,
                which="cos")
        
        cosine_modes.append(cos / speckle_amp * focal_amplitude)

        sin = dm.make_speckle_xy(
                xs=xi,
                ys=yi,
                amps=1,
                phases=0,
                centerx=xcen,
                centery=ycen,
                angle=dm_angle,
                lambdaoverd=lambdaoverd,
                N=60,
                which="sin")
        
        sine_modes.append(sin / speckle_amp * focal_amplitude)
        
        
    # Construct and scale the probes from the modes
    N_MODES = len(cosine_modes)
    rel_amplitude = amplitude # / N_MODES
    cosine_modes = np.asarray(cosine_modes) * rel_amplitude
    sine_modes = np.asarray(sine_modes) * rel_amplitude
    
    cos_probe = np.sum(cosine_modes, axis=0)
    sin_probe = np.sum(sine_modes, axis=0)

    # Take the probe measurements
    # NOTE: ref_img is the un-probed psf

    # apply DM shape
    current_dm_shape = AOSystem.get_dm_data()
    print(f"dm_shape rms before loop = {np.std(current_dm_shape)}")

    MAX_ITERS = settings['SN_SETTINGS']['NUM_ITERATIONS']

    hdu = fits.PrimaryHDU(ref_img)
    hdu.writeto(os.path.join(work_dir,"ref_img_Halfdark_ND1_5ms.fits"), overwrite=True)
    
    # clamp_val determines how well we can do, so we will regularize by this parameter as well
    clamp_val = sf.robust_sigma(ref_img[:50, :50].ravel())
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
    hdu.writeto(os.path.join(work_dir,"starting_dm_shape.fits"), overwrite=True)
    contrast_curves = []
    
    if AOSystem._closed:
        cos_probe = AOSystem.convert_voltage_to_cog(cos_probe)
        sin_probe = AOSystem.convert_voltage_to_cog(sin_probe)
    
    for i in range(MAX_ITERS):
        
        if i == 0:
            updated_dm_shape = current_dm_shape # + satspots
        
        if i != 0:
            ref_img = take_image_save_qacits(dir_name, i, timestamp, bgds)
            #ref_img = sf.equalize_image(Camera.take_image(), **bgds)

        vmin, vmax = np.percentile(ref_img, [0, 100])
        linthresh = 0.01 * max(np.abs(vmin), np.abs(vmax)) #before loop
        norm = SymLogNorm(vmin=vmin, vmax=vmax, linthresh=linthresh)
        if i == 0:
            im = ax[1].imshow(ref_img, norm=norm)
        else:
            im.set_data(ref_img)
        
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
        plt.pause(5)

        clamp_mask = clamp(ref_img, control_region, clamp=clamp_val)
        print(f"Pixels Clamped = {np.sum(1-clamp_mask)}")
        amp_mask = 1 # amplitude_weight(ref_img, control_region)

        # cosine probe
        cos_probe = cos_probe
        sin_probe = sin_probe

        print(f"cos probe rms = {np.std(cos_probe)}")
        print(f"sin probe rms = {np.std(sin_probe)}")
        print(f"dm_shape rms = {np.std(updated_dm_shape)}")
        set_shape = AOSystem.set_dm_data(updated_dm_shape + cos_probe)
        #cos_plus_img = sf.equalize_image(Camera.take_image(), **bgds)
        cos_plus_img = take_image_save_qacits(dir_name, f"iter{i}_cos_plus", timestamp, bgds) 
        
        # sine probe
        set_shape = AOSystem.set_dm_data(updated_dm_shape + sin_probe)
        #sin_plus_img = sf.equalize_image(Camera.take_image(), **bgds)
        sin_plus_img = take_image_save_qacits(dir_name, f"iter{i}_sin_plus", timestamp, bgds) 
        
        # cosine probe
        set_shape = AOSystem.set_dm_data(updated_dm_shape - cos_probe)
        #cos_minus_img = sf.equalize_image(Camera.take_image(), **bgds)
        cos_minus_img = take_image_save_qacits(dir_name, f"iter{i}_cos_minus", timestamp, bgds) 
        
        # sine probe
        set_shape = AOSystem.set_dm_data(updated_dm_shape - sin_probe)
        #sin_minus_img = sf.equalize_image(Camera.take_image(), **bgds)
        sin_minus_img = take_image_save_qacits(dir_name, f"iter{i}_sin_minus", timestamp, bgds) 
        
        im.set_data(sin_minus_img)
        plt.draw()
        plt.pause(1)
        
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

        # Compute the sine and cosine coefficients
        dE1 = (Ip1 - Im1) / 4
        dE2 = (Ip2 - Im2) / 4
        
        dE1sq = (Ip1 + Im1 - 2*I0) / 2
        dE2sq = (Ip2 + Im2 - 2*I0) / 2

        std_val = np.sqrt(I0)
        sin_coeffs = dE1 / (dE1sq + 1 * std_val) 
        cos_coeffs = dE2 / (dE2sq  + 1 * std_val)

        # sets nan or inf to zero
        sin_coeffs = condition_coeffs(sin_coeffs, maxval=2)
        cos_coeffs = condition_coeffs(cos_coeffs, maxval=2)
        
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
        
        control_surface = -1 * (sin_mode_control + cos_mode_control)
        control_surface -= np.mean(control_surface)
        
        # Apply correction and take image
        if AOSystem._closed:
            control_surface = AOSystem.convert_voltage_to_cog(control_surface)
        updated_dm_shape = updated_dm_shape + control_surface
        set_shape = AOSystem.set_dm_data(updated_dm_shape)
        corrected_img = sf.equalize_image(Camera.take_image(), **bgds)
        
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

        hdu = fits.PrimaryHDU(updated_dm_shape)
        hdu.writeto(os.path.join(work_dir,"SAN_iter{i}_dm_shape.fits"), overwrite=True)
        
        # write the images
        hdu = fits.PrimaryHDU(corrected_img)
        hdu.writeto(os.path.join(work_dir,f"SAN_iter{i}_Halfdark_ND1_5ms.fits"))


        intermediates = [I0, Im1, Ip1, Im2, Ip2]
        intermediates = np.asarray(intermediates)
        hdu = fits.PrimaryHDU(intermediates)
        hdu.header["IM1"] = "I0"
        hdu.header["IM2"] = "Im1 Sin"
        hdu.header["IM3"] = "Ip1 Sin"
        hdu.header["IM4"] = "Im2 Cos"
        hdu.header["IM5"] = "Ip2 Cos"
        hdu.writeto(os.path.join(work_dir,f"SAN_iter{i}_intermediates_ND1_5ms.fits"))
            
    # plt.ioff()
    # plt.close(fig)

    # Save Contrast Curves
    contrast_curves.append(np.ones_like(contrast_curves[0]) * clamp_val)
    contrast_curves = np.asarray(contrast_curves)
    hdu = fits.PrimaryHDU(contrast_curves)
    hdu.writeto(os.path.join(work_dir,"contrast_curves.fits"), overwrite=True)

    # FINAL: Return to flat
    set_shape = AOSystem.set_dm_data(current_dm_shape) 
