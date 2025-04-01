

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
    xi = 243
    yi = 135

    xi = 237
    yi = 173
    
    cos = dm.make_speckle_xy(
            xs=xi,
            ys=yi,
            amps=0.1,
            phases=0,
            centerx=xcen,
            centery=ycen,
            angle=dm_angle,
            lambdaoverd=lambdaoverd,
            N=21,
            which="cos")

    current_dm_shape = AOSystem.get_dm_data()
    cos_applied = AOSystem.set_dm_data(current_dm_shape + cos)
    probed_img = sf.equalize_image(Camera.take_image(), **bgds)
    return_flat = AOSystem.set_dm_data(current_dm_shape) # return to flat
    plt.figure()
    plt.imshow(probed_img, cmap="inferno", norm=LogNorm())
    plt.plot(xi, yi, marker="x", color="c")
    plt.show()
