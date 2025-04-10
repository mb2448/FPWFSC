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
## Libraries to handle speckle nulling files
import sn_filehandling  as flh 
## Libraries to import hardware used by speckle nulling
import sn_hardware as snh
## Libraries to create dm shapes
import dm_functions as dm
## Libraries use to perform some math operation specific to speckle nulling
import sn_math as snm
from detect_speckles import create_speckle_aperture, get_speckle_photometry, \
get_total_aperture_flux


from configobj import ConfigObj
import sn_filehandling as flh
#import flatmapfunctions as FM
from validate import Validator
#import flatmapfunctions_keck as fmf
import dm_functions as DM
import time
import scipy.ndimage as sciim
import scipy.optimize as opt 
import astropy.io.fits as pf
from detect_speckles import create_speckle_aperture, get_speckle_photometry, \
get_total_aperture_flux
############################## Local Definitions ##############################


     
if __name__ == "__main__":
    ''' -----------------------------------------------------------------------
    This program performs DM registration.
    It takes an image with the DM flat, then an image with speckles as defined
    in the configuration file provided by the user. Then reload initial DM map.
    It subtracts the two, asks you to click on the satellites and then figures
    out lambda/d, the center and rotation of the image.
    It then saves these values to the configuration file provided by the user.
    ----------------------------------------------------------------------- '''
    
    # Initialized config and spec file names
    soft_ini  = 'Config/SN_Software.ini'
    soft_spec = 'Config/SN_Software.spec'
    hard_ini  = 'Config/SN_Hardware.ini'
    hard_spec = 'Config/SN_Hardware.spec'
    
    # Read the config files and check if satisfied spec files requierements.
    soft_config = flh.validate_configfile(soft_ini, soft_spec)    

    # Hardware Connection
    print('')    
    print('############################################################')
    print('#################### Hardware Connection ###################')
    print('############################################################')
    print('')    

    # Instancie AO system and Detector selected
    AOsystem = snh.KECK2AO('KECKAO', hard_ini, hard_spec)
    Detector = snh.NIRC2(  'NIRC2' , hard_ini, hard_spec)

    # Beginning of the setup verification
    print('')    
    print('############################################################')
    print('################ Beginning coronagraph TA  #################')
    print('############################################################')
    print('')    

    # Read the detector calibration data. 
    # To acquire detector valibration data use the script:
    # Detector_Calibration.py
    print('Reading the calibration data acquired previously.')
    bgds = flh.setup_bgd_dict(soft_config)

    
    print('Get the inital shape of the DM.')
    # Get the DM shape currently apply to the DM    
    initial_dm_shape = AOsystem.get_dm_shape()

    print('Take a first image without speckles.')    
    # Acquire an image with the selected detector. 

    ap_rad = 8
    spotx = 154.2
    spoty = 175.4
    
    bgds = flh.setup_bgd_dict(soft_config)
    bgds['bkgd'] = None
    grad_history = []
    cost_history = []
    move_history = []
    gain = 0.8
    # Initial image
    im_0 = Detector.take_image()
    clean_image = pre.equalize_image(im_0, **bgds)
    aperture = create_speckle_aperture(clean_image, spotx, spoty, ap_rad)
    total_phot = get_total_aperture_flux(clean_image, aperture)
    J0 = np.log10(total_phot)
    cost_history.append(J0)
    # yank by a sub-pixel in the x direction
    delta_x = np.array([0.2,0])
    AOsystem.offset_FSM((delta_x)
    im_1 = Detector.take_image()
    clean_image = pre.equalize_image(im_1, **bgds)
    aperture = create_speckle_aperture(clean_image, spotx, spoty, ap_rad)
    total_phot = get_total_aperture_flux(clean_image, aperture)
    Jx = np.log10(total_phot)
    # yank by a sub-pixel in the y direction
    delta_y = np.array([0,0.2])
    AOsystem.offset_FSM(delta_y)
    im_2 = Detector.take_image()
    clean_image = pre.equalize_image(im_2, **bgds)
    aperture = create_speckle_aperture(clean_image, spotx, spoty, ap_rad)
    total_phot = get_total_aperture_flux(clean_image, aperture)
    Jy = np.log10(total_phot)
    # calculate gradient
    grad = np.asarray([(Jx - J0) / delta_x[0], (Jy - J0) / delta_y[1]])
    grad_history.append(grad)
    move_history.append(delta_x+delta_y)
    J0 = Jy
    cost_history.append(J0)
    for kk in range(0, 20):
        update = - gain * grad
        # x move in direction opposite to the gradient
        delta_x = np.array([update[0], 0])
        AOsystem.offset_FSM(delta_x)
        im_1 = Detector.take_image()
        clean_image = pre.equalize_image(im_1, **bgds)
        aperture = create_speckle_aperture(clean_image, spotx, spoty, ap_rad)
        total_phot = get_total_aperture_flux(clean_image, aperture)
        Jx = np.log10(total_phot)
        # yank by a sub-pixel in the y direction
        delta_y = np.array([0, update[1]])
        AOsystem.offset_FSM(delta_y)
        im_2 = Detector.take_image()
        clean_image = pre.equalize_image(im_2, **bgds)
        aperture = create_speckle_aperture(clean_image, spotx, spoty, ap_rad)
        total_phot = get_total_aperture_flux(clean_image, aperture)
        Jy = np.log10(total_phot)
        # calculate gradient
        grad = np.asarray([(Jx - J0) / delta_x[0], (Jy - J0) / delta_y[1]])
        grad_history.append(grad)
        move_history.append(delta_x + delta_y)
        J0 = Jy
        cost_history.append(J0)
        print('History of the cost function')
        print(cost_history)
        print('History of the relative moves')
        print(move_history)

    print('Closing stuff out')
    cost_historya = np.array(cost_history)
    min_cost = np.min(cost_historya)
    index_min_cost = np.where(cost_historya == min_cost)[0][0]
    if index_min_cost != len(cost_historya):
        print('Moving everything')
        move_back = - np.sum(move_historya[index_min_cost + 1:len(cost_historya)], 0)
        print(move_back)
        delta_x = np.array([move_back[0], move_back[1]])
        KeckSim.offset_FSM(delta_x)
        im_1 = KeckSim.take_image()
        clean_image = im_1
        w1.set_data(np.log(np.abs(clean_image)))
        plt.draw()
        plt.pause(0.02)
        aperture = create_speckle_aperture(clean_image, spotx, spoty, ap_rad)
        total_phot = get_total_aperture_flux(clean_image, aperture)
        J0 = np.log10(total_phot)
        cost_history.append(J0)
        move_history.append(delta_x)
        move_historya = np.array(move_history)
        w2 = ax2.plot(cost_history)
        w3 = ax3.plot(move_historya)
