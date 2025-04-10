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
# ## Function use to control NIRC2
# import Nirc2_cmds as Nirc2
# ## Function use to get the path where data should be saved
# from FIU_Commands import get_path

# # Location of the Speckle Nulling library
# sys.path.append('/kroot/src/kss/nirspec/nsfiu/dev/speckle_nulling/')
## Libraries to process speckle nulling data
import sn_preprocessing as pre
## Libraries to handle speckle nulling files
import sn_filehandling  as flh 
## Libraries to import hardware used by speckle nulling
#import sn_hardware as snh
import sn_sims as sns
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
    cal_waffle = DM.make_speckle_kxy(-10., 10.,  0.002, 0) + DM.make_speckle_kxy(10., 10., 0.002, 0)
    KeckSim.set_dm_shape(cal_waffle)
    KeckSim.make_aberration(0.08)
    KeckSim.make_TTM()

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




    ap_rad = 8
    spotx = 154.2
    spoty = 175.4
    
    bgds = flh.setup_bgd_dict(soft_config)
    bgds['bkgd'] = None
    grad_history = []
    cost_history = []
    move_history = []
    gain = 0.8
    # Offset the star in simulations
    x0 = np.array([4.0, -3.7])
    KeckSim.set_TTM(x0)
    # Initial image
    im_0 = KeckSim.take_image()
    #clean_image = pre.equalize_image(im_0, **bgds)
    clean_image = im_0
    aperture = create_speckle_aperture(clean_image, spotx, spoty, ap_rad)
    total_phot = get_total_aperture_flux(clean_image, aperture)
    J0 = np.log10(total_phot)
    # cost_history.append(J0)
    # yank by a sub-pixel in the x direction
    delta_x = np.array([-1.0,0])
    KeckSim.modulator_move(delta_x)
    im_1 = KeckSim.take_image()
    clean_image = im_1
    aperture = create_speckle_aperture(clean_image, spotx, spoty, ap_rad)
    total_phot = get_total_aperture_flux(clean_image, aperture)
    Jx = np.log10(total_phot)
    # yank by a sub-pixel in the y direction
    delta_y = np.array([0,-1.0])
    KeckSim.modulator_move(delta_y)
    im_2 = KeckSim.take_image()
    clean_image = im_2
    aperture = create_speckle_aperture(clean_image, spotx, spoty, ap_rad)
    total_phot = get_total_aperture_flux(clean_image, aperture)
    Jy = np.log10(total_phot)
    # calculate gradient
    grad = np.asarray([(Jx - J0) / delta_x[0], (Jy - J0) / delta_y[1]])
    grad_history.append(grad)
    move_history.append(delta_x+delta_y)
    J0 = Jy
    cost_history.append(J0)

    plt.ion()
    fig = plt.figure(figsize=(10, 4))
    ax1 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1)
    title = fig.suptitle('Target Acquisition')
    ax1.set_title('Image')
    ax2.set_title('Cost function')
    ax3.set_title('Move history in x and y')
    w1 = ax1.imshow(np.log(np.abs(clean_image)), origin='lower', interpolation='nearest')
    w2 = ax2.plot(cost_history)
    move_historya = np.array(move_history)
    w3 = ax3.plot(move_historya)
    plt.draw()
    plt.pause(0.02)




    print('History of the cost function')
    print(cost_history)
    print('History of the relative moves')
    print(move_history)
    for kk in range(0, 40):
        print(kk)
        update = - gain * grad
        # x move in direction opposite to the gradient
        delta_x = np.array([update[0], 0])
        KeckSim.modulator_move(delta_x)
        im_1 = KeckSim.take_image()
        clean_image = im_1
        w1.set_data(np.log(np.abs(clean_image)))
        plt.draw()
        plt.pause(0.02)
        aperture = create_speckle_aperture(clean_image, spotx, spoty, ap_rad)
        total_phot = get_total_aperture_flux(clean_image, aperture)
        Jx = np.log10(total_phot)
        #  x move in direction opposite to the gradient
        delta_y = np.array([0, update[1]])
        KeckSim.modulator_move(delta_y)
        im_2 = KeckSim.take_image()
        clean_image = im_2
        w1.set_data(np.log(np.abs(clean_image)))
        plt.draw()
        plt.pause(0.02)
        aperture = create_speckle_aperture(clean_image, spotx, spoty, ap_rad)
        total_phot = get_total_aperture_flux(clean_image, aperture)
        Jy = np.log10(total_phot)
        # calculate gradient
        print(delta_x[0])
        print(delta_y[1])
        grad = np.asarray([(Jx - J0) / delta_x[0], (Jy - J0) / delta_y[1]])
        grad_history.append(grad)
        move_history.append(delta_x + delta_y)
        move_historya = np.array(move_history)
        J0 = Jy
        cost_history.append(J0)
        w2 = ax2.plot(cost_history)
        w3 = ax3.plot(move_historya)
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
        move_back =   - np.sum(move_historya[index_min_cost+1:len(cost_historya)],0)
        print(move_back)
        delta_x = np.array([move_back[0], move_back[1]])
        KeckSim.modulator_move(delta_x)
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






