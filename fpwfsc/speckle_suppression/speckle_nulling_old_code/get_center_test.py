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

if __name__ == "__main__":
    #nospot_image = fits.open('no_waffle.fits')[0].data
    #spot_image = fits.open('with_waffle.fits')[0].data
    
    # Initialized config and spec file names
    soft_ini  = 'Config/SN_Software.ini'
    soft_spec = 'Config/SN_Software.spec'
    hard_ini  = 'Config/SN_Hardware.ini'
    hard_spec = 'Config/SN_Hardware.spec'

    AOsystem = snh.KECK2AO('KECKAO', hard_ini, hard_spec)
    Detector = snh.NIRC2(  'NIRC2' , hard_ini, hard_spec)
    
    soft_config = flh.validate_configfile(soft_ini, soft_spec)    
    bgds = flh.setup_bgd_dict(soft_config)
    xc = soft_config['IM_PARAMS']['centerx']
    yc = soft_config['IM_PARAMS']['centery']
    
    set_x = soft_config['QACITS']['setpointx'] 
    set_y = soft_config['QACITS']['setpointy'] 
    setpoint = np.array([set_x, set_y]) 
    plt.ion()    
    fig, ax = plt.subplots()
    x, y = [], []
    sc_setpoint = ax.scatter([set_x],[set_y], color = 'red',alpha = 1.0)
    sc = ax.scatter(x,y, alpha = 0.5)
    plt.xlim(xc-10, xc+10)    
    plt.ylim(yc-10, yc+10)    
    plt.draw()
    while True:
        im_0 = Detector.take_image() ###
        clean_image = pre.equalize_image(im_0, **bgds) ###
        center, intensity = get_satellite_center_and_intensity(clean_image, soft_ini, soft_spec)###
        x.append(center[0])
        y.append(center[1])
        print("Intensity: ", intensity)
        print("\n\n")
        print("Setpoint: ", setpoint)
        print("Center: ",center)
        sc.set_offsets(np.c_[x,y])
        fig.canvas.draw_idle()
        plt.pause(0.05)
        error = center-setpoint
        print("Error: ", error)
        gain = 0.85
        AOsystem.TTM_move(-1*error*gain)
    plt.waitforbuttonpress()
