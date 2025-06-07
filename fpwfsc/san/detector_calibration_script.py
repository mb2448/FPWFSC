import numpy as np
import os
import sys
import astropy.io.fits as fits
from collections import deque
import threading

import fpwfsc.common.support_functions as sf
import fpwfsc.common.bench_hardware as hw

def build_master_flat(data, badpix=None, kern=9, removezeros=True):
    '''Removes bad pixels from a background subtracted master flat'''
    MF = sf.removebadpix(data, badpix, kern=kern)
    MF = MF/np.mean(MF)
    if removezeros: MF = sf.removebadpix(MF, MF == 0, kern=kern)
    return MF

def get_user_input(prompt, valid_options=None, default=None):
    """Standardized user input function with validation"""
    while True:
        answer = input(prompt)
        if not answer and default is not None:
            return default
        if valid_options is None or answer.lower() in valid_options:
            return answer.lower()
        print(f"Invalid input. Valid options are: {valid_options}")

def run(camera=None, aosystem=None, config=None, configspec=None,
        my_deque=None, my_event=None, plotter=None):
    # Initialize threading objects if needed
    if my_deque is None: my_deque = deque()
    if my_event is None: my_event = threading.Event()

    # Load configuration
    settings = sf.validate_config(config, configspec)

    # Setup instruments
    if camera == 'Sim' and aosystem == 'Sim':
        import fpwfsc.common.fake_hardware as fhw
        CSM = fhw.FakeCoronagraphOpticalSystem(**settings['SIMULATION']['OPTICAL_PARAMS'])
        AOSystem = fhw.FakeAODMSystem(OpticalModel=CSM, **settings['SIMULATION']['AO_PARAMS'])
        Camera = fhw.FakeDetector(opticalsystem=CSM, **settings['SIMULATION']['CAMERA_PARAMS'])
    else:
        Camera = camera
        AOSystem = aosystem

    # Create output directory
    outputdir = settings['CAMERA_CALIBRATION']['bgddir']
    os.makedirs(outputdir, exist_ok=True)

    # Take default image to get dimensions
    print("Taking default image")
    default_image = Camera.take_image()

    # System verification
    print('\n' + '#'*60 + '\n############### System Settings Verification ###############\n' + '#'*60)
    if get_user_input('\nCheck system settings? (i/enter): ', ['i', '']) == 'i':
        print("TODO: XYZ FIX THIS THESE DONT EXIST")
        # Camera.update_parameters()
        # Camera.print_parameters()

    # Get number of images for calibration
    print('\n' + '#'*60 + '\n###### Number of images acquired for calibration data ######\n' + '#'*60)
    nb_images = 3
    try:
        input_val = input(f" {nb_images:02d} images will be acquired for each calibration data.\n"
                         "To modify (1-20), enter a number, or press enter to continue: ")
        if input_val:
            tmp = int(input_val)
            if 0 < tmp < 21:
                nb_images = tmp
                print(f"Number of images set to {nb_images:02d}")
            else:
                print("Invalid value. Using default.")
    except ValueError:
        print("Invalid input. Using default.")

    # Define calibration types with their parameters
    calibration_types = [
        {
            "name": "Background",
            "default_desc": "Array of 0",
            "default_value": np.zeros(default_image.shape),
            "filename": "medbackground.fits"
        },
        {
            "name": "Flat",
            "default_desc": "array of 1",
            "default_value": np.ones(default_image.shape),
            "filename": "medflat.fits"
        },
        {
            "name": "Flat Dark",
            "default_desc": "array of 0",
            "default_value": np.zeros(default_image.shape),
            "filename": "medflatdark.fits"
        }
    ]

    # Acquire all calibration data
    calibration_results = {}

    for cal_type in calibration_types:
        name = cal_type["name"]
        print(f'\n{"#"*60}\n################### {name} Acquisition ###################\n{"#"*60}\n')

        user_choice = get_user_input(
            f'\nUse default {name.lower()} ({cal_type["default_desc"]}), skip this step, or acquire new data? (d/s/enter): ',
            ['d', 's', '']
        )

        if user_choice == 's':
            print(f'Skipping {name.lower()} acquisition')
            continue  # Skip to the next calibration type
        elif user_choice == 'd':
            print(f'Using default {name.lower()} ({cal_type["default_desc"]})')
            data = cal_type["default_value"]
        else:
            print(f'Acquiring {name.lower()}')

            if get_user_input(f'\nIs system ready for {name.lower()} acquisition? (i/enter): ',
                             ['i', '']) == 'i':
                print("TODO: XYZ FIX THIS THESE DONT EXIST")
                # Camera.update_parameters()
                # Camera.print_parameters()

            # Acquire and process images
            data_cube = np.zeros([nb_images] + list(default_image.shape))
            for i in range(nb_images):
                data_cube[i] = Camera.take_image()
            data = np.median(data_cube, 0)

        # Save the data
        output_path = os.path.join(outputdir, cal_type["filename"])
        fits.PrimaryHDU(data).writeto(output_path, overwrite=True)
        os.chmod(output_path, 0o644)
        print(f'The {name.lower()} has been saved to: {output_path}')

        # Store for later processing
        calibration_results[name.lower()] = data

    # Process bad pixel map
    print('\n' + '#'*60 + '\n####################### Bad Pixel Map ######################\n' + '#'*60 + '\n')

    flatdark = calibration_results["flat dark"]
    if np.allclose(flatdark, 0):
        badpix = np.copy(flatdark)
        print('Using default bad pixel map (array of 0)')
    else:
        badpix = sf.locate_badpix(flatdark, sigmaclip=3)
        print('Generated bad pixel map from flat dark')

    badpix_path = os.path.join(outputdir, 'badpix.fits')
    fits.PrimaryHDU(badpix).writeto(badpix_path, overwrite=True)
    print(f'Bad pixel map saved to: {badpix_path}')

    # Process master flat
    print('\n' + '#'*60 + '\n###################### Master Flat Map #####################\n' + '#'*60 + '\n')

    flat = calibration_results["flat"]
    if np.allclose(flat, 1):
        masterflat = np.copy(flat)
        print('Using default master flat (array of 1)')
    else:
        masterflat = build_master_flat(flat - flatdark, badpix=badpix)
        print('Generated master flat from flat and flat dark')

    masterflat_path = os.path.join(outputdir, 'masterflat.fits')
    fits.PrimaryHDU(masterflat).writeto(masterflat_path, overwrite=True)
    print(f'Master flat saved to: {masterflat_path}')

    print('\n' + '#'*60 + '\n#################### Return to operation ###################\n' + '#'*60 + '\n')
    print('Do not forget to set up the system properly.')
    return

if __name__ == "__main__":
    # Camera = hw.NIRC2Alias()
    # AOSystem = hw.AOSystemAlias()
    # # Camera, AOSystem = 'Sim', 'Sim'
    # run(camera=Camera, aosystem=AOSystem, config='fpwfsc/san/sn_config.ini', configspec='fpwfsc/san/sn_config.spec')
    Camera = hw.NIRC2Alias()
    AOSystem = hw.AOSystemAlias()
    # Camera, AOSystem = 'Sim', 'Sim'
    run(camera=Camera, aosystem=AOSystem, config='sn_config.ini', configspec='sn_config.spec')
