############################### Import Library ################################

## Math Library
import numpy as np
## System library
import sys
## Operating system library
import os
## .fits file library
import astropy.io.fits as fits
## Library used to plot graphics and show images
import matplotlib.pyplot as plt
# Location of the FIU library
sys.path.append('/kroot/src/kss/nirspec/nsfiu/dev/lib/')
## Function use to control NIRC2
import Nirc2_cmds as Nirc2
## Function use to get the path where data should be saved
from FIU_Commands import get_path

# Location of the Speckle Nulling library
sys.path.append('/kroot/src/kss/nirspec/nsfiu/dev/speckle_nulling/')
## Libraries to process speckle nulling data
import sn_preprocessing as pre
import sn_filehandling as flh
from dm_registration import get_satellite_centroids, \
get_satellite_center_and_intensity
import sn_hardware as snh
import time
if __name__ == "__main__":
    nospot_image = fits.open('no_waffle.fits')[0].data
    spot_image = fits.open('with_waffle.fits')[0].data
    
    # Initialized config and spec file names
    soft_ini  = 'Config/SN_Software.ini'
    soft_spec = 'Config/SN_Software.spec'
    hard_ini  = 'Config/SN_Hardware.ini'
    hard_spec = 'Config/SN_Hardware.spec'
    
    soft_config = flh.validate_configfile(soft_ini, soft_spec)    
    bgds = flh.setup_bgd_dict(soft_config)
    xc = soft_config['IM_PARAMS']['centerx']
    yc = soft_config['IM_PARAMS']['centery']
    
    im_0 = spot_image
    clean_image = pre.equalize_image(im_0, **bgds)
    time0 = time.time()
    center, intensity = get_satellite_center_and_intensity(clean_image, 
                                                soft_ini, soft_spec)
    time1 = time.time()
    print("Center: ", center)
    print("Intensity: ", intensity)
