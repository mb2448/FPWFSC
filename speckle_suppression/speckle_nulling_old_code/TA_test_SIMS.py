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
# sys.path.append('/kroot/src/kss/nirspec/nsfiu/dev/lib/')
## Function use to control NIRC2
# import Nirc2_cmds as Nirc2
## Function use to get the path where data should be saved
# from FIU_Commands import get_path

# Location of the Speckle Nulling library
#sys.path.append('/kroot/src/kss/nirspec/nsfiu/dev/speckle_nulling/')
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
    This program performs Target Acquisition. 
    It starts with the star somewhere close to the coronagraph mask, 
    and uses gradient descent on the encircled energy to aliggn the star under the vortex.
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
    # AOsystem = snh.KECK2AO('KECKAO', hard_ini, hard_spec)
    # Detector = snh.NIRC2(  'NIRC2' , hard_ini, hard_spec)
    KeckSim = sns.FakeCoronagraph()
    KeckSim.make_DM()
    # speckle_x0 = DM.make_speckle_kxy(4.,0., 0.1, 0)
    # abb0 = speckle_x0
    # KeckSim.set_dm_shape(abb0)
    cal_waffle = DM.make_speckle_kxy(-10., 10., 0.005, 0) + DM.make_speckle_kxy(10., 10., 0.005, 0)
    KeckSim.set_dm_shape(cal_waffle)
    KeckSim.make_aberration(0.1)

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

    ap_rad = 20
    spotx = 154.2
    spoty = 175.4
    
    bgds = flh.setup_bgd_dict(soft_config)
    bgds['bkgd'] = None
    grad_history = []
    cost_history = []
    move_history = []
    gain = 1
    for kk in range(0, 10):
        print(kk)
        im_0 = Detector.take_image()
        clean_image = pre.equalize_image(im_0, **bgds)
        clean_image[clean_image<0] = np.max(clean_image) #hmmmmmm
        aperture = create_speckle_aperture(clean_image,spotx, spoty,ap_rad)  
        total_phot = get_total_aperture_flux(clean_image, aperture)
        cost_TA_0 = np.log10(total_phot)
        orient_wiggle = np.random.uniform(0, 2 * np.pi, 1)[0]
        delta_x = np.array([np.cos(orient_wiggle), np.sin(orient_wiggle)])
        AOsystem.modulator_move(delta_x)
        im_1 = Detector.take_image()
        clean_image = pre.equalize_image(im_1, **bgds)
        clean_image[clean_image<0] = np.max(clean_image) #hmmmmmm
        aperture = create_speckle_aperture(clean_image,spotx, spoty,ap_rad)  
        total_phot = get_total_aperture_flux(clean_image, aperture)
        cost_TA_1 = np.log10(total_phot)
        grad = np.asarray([(cost_TA_1 - cost_TA_0) / (delta_x[0]), (cost_TA_1 - cost_TA_0) / (delta_x[1])])
        print('gradient =', grad)
        grad = grad / np.linalg.norm(grad)
        delta_x_move= delta_x/2 - gain * grad
        print('Actual TT move sent =',delta_x_move) 
        AOsystem.modulator_move(delta_x_move)
        grad_history.append(grad)
        cost_history.append(cost_TA_0)
        move_history.append(delta_x_move)
        print('History of the cost function')
        print(cost_history)
