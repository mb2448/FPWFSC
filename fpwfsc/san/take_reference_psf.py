
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

if __name__ == "__main__":
    
    # User options are pulled from sn_config.ini
    # make timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"reference_psf"
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
    src_destination = os.path.join(dir_name, src)
    shutil.copy(folder+src, src_destination)
    
    src = configspec
    src_destination = os.path.join(dir_name, src)
    shutil.copy(folder+src, src_destination)

    src = "take_reference_psf.py"
    src_destination = os.path.join(dir_name, src)
    shutil.copy(folder+src, src_destination)

    # Save the Backgrounds
    # src = settings['CAMERA_CALIBRATION']['bgddir']
    # dst = dir_name
    # shutil.copytree(src, os.path.join(dst, "bgds"))

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

    #Ref PSF settings
    tint                = settings['SN_SETTINGS']['TINT']
    coadds              = settings['SN_SETTINGS']['COADDS']
    nd_filter           = settings['SN_SETTINGS']['FILTER']

    tint = float(tint)
    coadds = float(coadds)
    nd_filter = float(nd_filter)
    xcen = int(xcen)
    ycen = int(ycen)

    print(tint, coadds, nd_filter, xcen, ycen)

    # convert IWA/OWA to pixels
    IWA_pix = IWA * lambdaoverd
    OWA_pix = OWA * lambdaoverd 
    ref_img = sf.equalize_image(Camera.take_image(), **bgds)
    ref_img *= nd_filter / (tint * coadds) # get to counts per second

    dxrad = 5
    contrast_ref = ref_img[ycen-dxrad:ycen+dxrad, xcen-dxrad:xcen+dxrad].mean()
    contrast_ref = ref_img.max()
    settings['CONTRAST_CALIBRATION']['reference_flux'] = contrast_ref
    config_file = folder + config

    # Save contrast reference to config file
    print("Saving reference flux to: ", config_file)
    with open(config_file, 'wb') as configfile:
        settings.write(configfile)

    # Save reference PSF to fits file
    hdu = fits.PrimaryHDU(ref_img)
    hdu.writeto(os.path.join(dir_name,"ref_psf.fits"), overwrite=True)


