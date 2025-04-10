############################### Import Library ################################

## Math Library
import numpy as np
## System library
import sys
## Operating system library
import os
## .fits file library
import astropy.io.fits as fits

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

############################## Local Definitions ##############################

# =============================================================================
def build_master_flat(data, badpix=None, kern = 9, removezeros = True):
    ''' -----------------------------------------------------------------------
    Removes bad pixels from a background subtracted master flat
    ----------------------------------------------------------------------- '''
    # Remove bad pixel from the provided background substracted flat
    MF = pre.removebadpix(data, badpix, kern=kern)
    # Normalize the masterflat 
    MF = MF/np.mean(MF)
    # Replace the zeros by the associated medfilt value kern x kern
    if removezeros: MF = pre.removebadpix(MF,MF == 0, kern = kern)
    # Return the master_flat
    return MF

# =============================================================================
if __name__ == "__main__":
    ''' -----------------------------------------------------------------------
    This script is meant to acquire with the selected detector backgrounds, 
    darks and flats, then assele them into the correctly formatted region that  
    we care about, and place them into:
        - today '/Speckle_Nulling/Calibrations_Data/' directory.
    ----------------------------------------------------------------------- '''

    # Get Path where calibration data will be saved 
    Path, tmp = get_path('/Speckle_Nulling/Calibrations_Data/Detector/')

    # Initialized config and spec file names
    soft_ini  = 'Config/SN_Software.ini'
    soft_spec = 'Config/SN_Software.spec'
    hard_ini  = 'Config/SN_Hardware.ini'
    hard_spec = 'Config/SN_Hardware.spec'

    # Read the config files and check if satisfied spec files requierements.
    soft_config = flh.validate_configfile(soft_ini, soft_spec)

    # Extract from the software init file the directory where data must be 
    # saved and the number of image required to built the dark
    outputdir = soft_config['DETECTOR_CAL']['dirwr']
    nb_images = soft_config['DETECTOR_CAL']['nb_im']

    print('')    
    print('############################################################')
    print('#################### Hardware Connection ###################')
    print('############################################################')
    print('')    

    # Instancie the selected detector
    Detector = snh.NIRC2('NIRC2', hard_ini, hard_spec) 
    
    print("Taking default image")
    default_image = Detector.take_image()
    sizex, sizey = default_image.shape
    # Display the relevant Detector parameters
    Detector.print_parameters()

    # Beginning of the setup verification
    print('')    
    print('############################################################')
    print('############### System Settings Verification ###############')
    print('############################################################')
    
    # Prepare message for user
    msg  = '''\nAre the system settings valid?                        ''' 
    msg += '''\nTo read and display relevant parameters:              '''
    msg += '''\n    - press 'I' then 'enter'.                         '''
    msg += '''\nTo go to the next step:                               '''
    msg += '''\n    - press 'enter'.                                  '''

    # Flag for the following while loop status
    Flag = True
    # Enter in a while loop
    while Flag:
        # Check with user if detector settings are correct
        answer= input(msg)
        if answer.lower() == 'i':
            # Read relevant detector parameters
            Detector.update_parameters()
            # Display the current detector parameters
            Detector.print_parameters()
        else:
            # Exit the while loop by changing Flag value
            Flag = False
    # Define the number of image to average for the calibration data
    print('')    
    print('############################################################')
    print('###### Number of images acquired for calibration data ######')
    print('############################################################')
    
    # Prepare a message for the user
    msg  = ''' %02d '''  %(nb_images) 
    msg += ''' images will be acquired for each calibration data.     '''
    msg += '''\nTo modify this number:                                '''
    msg += '''\n    - type an integer between [1 and 20],             '''
    msg += '''\n    - press 'enter'.                                  '''
    msg += '''\nTo go to the next step:                               '''
    msg += '''\n    - press 'enter'.                                  '''
    # Check if the user want to modify the number of images used to
    # compute the background
    answer= input(msg)

    # Check if the user enter a new value
    if answer != '':
        # if a new value has been enter check if it is a number
        try:
            tmp = np.int(answer)
            # Check if the number of image is valid
            if 0 < tmp < 21:
                # Update the number of images to use for the background
                nb_images = tmp 
                # Prepare a message for the user
                msg  = '''The number of images has been modified.     '''
                msg += '''\nThe calibration data will be a median of  '''
                msg += '''%02d images.''' %(nb_images)
                print(msg)
            else:
                print('The value enter is not valid.')
                print('The number of images has not been modified.')
        except:
            tmp = ''
    else:
        print('The number of images has not been modified.')

    print('')    
    print('############################################################')
    print('################### Bacground Acquisition ##################')
    print('############################################################')
    print('') 
    # Prepare message for user
    msg  = '''\nWould you like to use a default bkgd (Array of 0)?    '''
    msg += '''\n    - if yes, type 'D' then press 'enter'.            '''
    msg += '''\n    - if not, press 'enter'                           '''

    # Check if user want to use a default background
    answer= input(msg)

    if answer.lower() == 'd':
        # Prepare a message for the user
        msg = '''You decided to use a default background (Array of 0).'''
        # Print message
        print(msg)
        # Generate an array of zeros used as a default background 
        #bkgd = np.zeros([sizex,sizey])
        bkgd = np.zeros(default_image.shape)
    else:
        # Prepare a message for the user
        msg = '''You decided to acquire a background.                '''
        # Print message
        print(msg)

        # Prepare message for user
        msg  = '''\nIs the system ready for background acquisition?  ''' 
        msg += '''\nTo read and display relevant parameters:         '''
        msg += '''\n    - press 'I' then 'enter'.                    '''
        msg += '''\nTo go to the next step:                          '''
        msg += '''\n    - press 'enter'.                             '''

        # Flag for the following while loop status
        Flag = True
        # Enter in a while loop
        while Flag:
            # Check with user if detector settings are correct
            answer= input(msg)
            if answer.lower() == 'i':
                # Read all detector parameters
                Detector.update_parameters()
                # Display the current Nirc2 parameters
                Detector.print_parameters()
            else:
                # Exit the while loop by changing Flag value
                Flag = False
        
        # Initialize bkgd cube of images
        bkgd = np.zeros([nb_images,sizex,sizey])
        # Background acquisition with the selected detector
        for i in np.arange(nb_images): bkgd[i,:,:] = Detector.take_image()
        # Compute the median of the images acquired
        bkgd = np.median(bkgd,0)

    # Save the background (Mike style)
    hdu = fits.PrimaryHDU(bkgd)
    hdu.writeto(os.path.join(outputdir,'medbackground.fits'),overwrite = True)

    # Save the background (FIU style)
    hdu = fits.PrimaryHDU(bkgd)
    hdu.writeto(Path + 'medbackground.fits',overwrite = True)

    # Print message for user
    print('The bad pixel map has been saved to this/these location(s):')
    print('    - ' + os.path.join(outputdir,'medbackground.fits'))
    print('    - ' + Path + 'medbackground.fits')        

    print('')    
    print('############################################################')
    print('##################### Flat Acquisition #####################')
    print('############################################################')
    print('') 
    # Prepare message for user
    msg  = '''\nWould you like to use a default flat (array of 1)?    '''
    msg += '''\n    - if yes, type 'D' then press 'enter'.            '''
    msg += '''\n    - if not, press 'enter'                           '''

    # Check if user want to use a default background
    answer= input(msg)

    if answer.lower() == 'd':
        # Prepare a message for the user
        msg = '''You decided to use a default flat (array of 1).      '''
        # Print message
        print(msg)
        # Generate an array of zeros used as a default background 
        flat = np.ones(default_image.shape)
    else:
        # Prepare a message for the user
        msg = '''You decided to generate a flat.                      '''
        # Print message
        print(msg)
        
        # Prepare message for user
        msg  = '''\nIs the detector ready for flat acquisition?      ''' 
        msg += '''\nTo read and display Relevant parameters:         '''
        msg += '''\n    - press 'I' then 'enter'.                    '''
        msg += '''\nTo go to the next step:                          '''
        msg += '''\n    - press 'enter'.                             '''

        # Flag for the following while loop status
        Flag = True
        # Enter in a while loop
        while Flag:
            # Check with user if the detector settings are correct
            answer= input(msg)
            if answer.lower() == 'i':
                # Read all the detector parameters
                Detector.update_parameters()
                # Display the current the detector parameters
                Detector.print_parameters()
            else:
                # Exit the while loop by changing Flag value
                Flag = False

        # Initialize flat cube of images
        flat = np.zeros([nb_images,sizex,sizey])
        # Flat acquisition with the selected detector
        for i in np.arange(nb_images): flat[i,:,:] = Detector.take_image()
        # Compute the median of the images acquired
        flat = np.median(flat,0)

    # Save the flat (Mike style)
    hdu = fits.PrimaryHDU(flat)
    hdu.writeto(os.path.join(outputdir,'medflat.fits'),overwrite = True)

    # Save the flat (FIU style)
    hdu = fits.PrimaryHDU(flat)
    hdu.writeto(Path + 'medflat.fits',overwrite = True)

    # Print message for user
    print('The flat has been saved to this/these location(s):')
    print('    - ' + os.path.join(outputdir,'medflat.fits'))
    print('    - ' + Path + 'medflat.fits')        
    
    print('')    
    print('############################################################')
    print('################### Flat Dark Acquisition ##################')
    print('############################################################')
    print('') 
    # Prepare message for user
    msg  = '''\nWould you like to use a default flat dark(array of 0)?'''
    msg += '''\n    - if yes, type 'D' then press 'enter'.            '''
    msg += '''\n    - if not, press 'enter'                           '''

    # Check if user want to use a default flatdark
    answer= input(msg)

    if answer.lower() == 'd':
        # Prepare a message for the user
        msg = '''You decided to use a default flat dark (array of 0).  '''
        # Print message
        print(msg)
        # Generate an array of zeros used as a default flatdark
        flatdark = np.zeros([sizex,sizey])
    else:
        # Prepare a message for the user
        msg = '''You decided to generate a flat dark.                 '''
        # Print message
        print(msg)
        
        # Prepare message for user
        msg  = '''\nIs the system ready for flat dark  acquisition?  ''' 
        msg += '''\nTo read and display relevant system parameters:  '''
        msg += '''\n    - press 'I' then 'enter'.                    '''
        msg += '''\nTo go to the next step:                          '''
        msg += '''\n    - press 'enter'.                             '''

        # Flag for the following while loop status
        Flag = True
        # Enter in a while loop
        while Flag:
            # Check with user if the camera settings are correct
            answer= input(msg)
            if answer.lower() == 'i':
                # Read all detector parameters
                Detector.update_parameters()
                # Display the current detector parameters
                Detector.print_parameters()
            else:
                # Exit the while loop by changing Flag value
                Flag = False

        # Initialize flatdark cube of images
        flatdark = np.zeros([nb_images,sizex,sizey])
        # Flatdark acquisition with the selected detector
        for i in np.arange(nb_images): flatdark[i,:,:] = Detector.take_image()
        # Compute the median of the images acquired
        flatdark = np.median(flatdark,0)

    # Save the flatdark (Mike style)
    hdu = fits.PrimaryHDU(flatdark)
    hdu.writeto(os.path.join(outputdir,'medflatdark.fits'),overwrite = True)

    # Save the flatdark (FIU style)
    hdu = fits.PrimaryHDU(flatdark)
    hdu.writeto(Path + 'medflatdark.fits',overwrite = True)

    ## Print message for user
    #print('The flat dark has been saved to this/these location(s):')
    #print('    - ' + os.path.join(outputdir,'medflatdark.fits'))
    #print('    - ' + Path + 'medflatdark.fits')

 
    print('')    
    print('############################################################')
    print('####################### Bad Pixel Map ######################')
    print('############################################################')
    print('')    
    # Check if the flatdark is a default one (array of 0)
    if np.allclose(flatdark, 0):
        # Generate a default masterflat
        badpix = np.copy(flatdark)
        # Print message for user
        print('A default flatdark has been generated (array of 0)')
    else:
        # Generate a  Badpixel map from the darks acquired
        badpix = pre.locate_badpix(flatdark, sigmaclip = 3)
        # Print message for user
        print('A master flat has been generated from the dark.')

    # Save the badpix (Mike style)
    hdu = fits.PrimaryHDU(badpix)
    hdu.writeto(os.path.join(outputdir,'badpix.fits'),overwrite = True)
    
    # Save the flatdark (FIU style)
    hdu = fits.PrimaryHDU(badpix)
    hdu.writeto(Path + 'badpix.fits',overwrite = True)
    
    ## Print message for user
    #print('The bad pixel map has been saved to this/these location(s):')
    #print('    - ' + os.path.join(outputdir,'badpix.fits'))
    #print('    - ' + Path + 'badpix.fits')        

    print('')    
    print('############################################################')
    print('###################### Master Flat Map #####################')
    print('############################################################')
    print('')    
    # Check if the masterflat is a default one (array of 1)
    if np.allclose(flat, 1):
        # Generate a default masterflat
        masterflat = np.copy(flat)
        # Print message for user
        print('A default master flat has been generated (array of 1)')
    else:
        # Generate a  masterflat from the flats and darks acquired
        masterflat = build_master_flat(flat-flatdark, badpix=badpix)
        # Print message for user
        print('A master flat has been generated from the flat and dark.')

    # Save the masterflat (Mike style)
    hdu = fits.PrimaryHDU(masterflat)
    hdu.writeto(os.path.join(outputdir,'masterflat.fits'),overwrite = True)
    
    # Save the masterflat (FIU style)
    hdu = fits.PrimaryHDU(masterflat)
    hdu.writeto(Path + 'masterflat.fits',overwrite = True)

    ## Print message for user
    #print('The master flat has been saved to this/these location(s):')
    #print('    - ' + os.path.join(outputdir,'masterflat.fits'))
    #print('    - ' + Path + 'masterflat.fits')

    print('')    
    print('############################################################')
    print('#################### Return to operation ###################')
    print('############################################################')
    print('') 
    # Print message for user
    print('Do not forget to set up the system properly.')
    
    
