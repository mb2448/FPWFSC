import numpy as np
import sys
import threading
import time
from collections import deque
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from fpwfsc.san import qt_clickpoints
from fpwfsc.san import sn_functions as sn

from fpwfsc.common import dm
from fpwfsc.common import support_functions as sf
from fpwfsc.common import fake_hardware as fhw
from fpwfsc.common import bench_hardware as hw


import ipdb

def run(camera=None, aosystem=None, config=None, configspec=None,
        my_deque=None, my_event=None, plotter=None):
    if my_deque is None:
        my_deque = deque()

    if my_event is None:
        my_event = threading.Event()
    
    # Store the config filename for later use
    config_file = config
    
    settings = sf.validate_config(config, configspec)
    #----------------------------------------------------------------------
    # Simulation parameters
    # Default settings 
    #----------------------------------------------------------------------
    
    #SN Settings
    xcen                = settings['SN_SETTINGS']['xcen']
    ycen                = settings['SN_SETTINGS']['ycen']
    cropsize            = settings['SN_SETTINGS']['cropsize']   
    #----------------------------------------------------------------------
    # Load instruments
    #----------------------------------------------------------------------
    if camera == 'Sim' and aosystem == 'Sim':
        #----------------------------------------------------------------------
        #CAMERA, AO, CORONAGRAPH SIM SETTINGS IN CONFIG FILE
        #----------------------------------------------------------------------
        CSM      = fhw.FakeCoronagraphOpticalSystem(**settings['SIMULATION']['OPTICAL_PARAMS'])
        AOSystem = fhw.FakeAODMSystem(OpticalModel=CSM, **settings['SIMULATION']['AO_PARAMS'])
        Camera   = fhw.FakeDetector(opticalsystem=CSM, **settings['SIMULATION']['CAMERA_PARAMS'])
    
    else:
        Camera = camera #hw.Camera.instance()
        AOSystem = aosystem #hw.AOSystem.instance()

    bgds = sf.setup_bgd_dict(settings['CAMERA_CALIBRATION']['bgddir'])
    
    # Now proceed with the intensity calibration
    intconf = settings['DM_REGISTRATION']['INTENSITY_CAL']
    imageparams = settings['DM_REGISTRATION']['MEASURED_PARAMS']

    #take bgd image with no speckles
    data_nospeck_raw = Camera.take_image()
    data_nospeck = sf.equalize_image(data_nospeck_raw, **bgds)
    xypixels = []
    ximcoords, yimcoords = np.meshgrid(np.arange(data_nospeck.shape[0]),
                                      np.arange(data_nospeck.shape[1]))

    # Create a non-blocking viewer instead of the blocking one
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create the non-blocking viewer using our new function
    viewer = qt_clickpoints.create_non_blocking_viewer(data=(np.abs(data_nospeck+1)))
    
    # Set the DeleteOnClose attribute to ensure proper cleanup
    viewer.setAttribute(Qt.WA_DeleteOnClose)
    
    kr = np.arange(intconf['min'],
                   intconf['max'],
                   intconf['stepsize'])
    DMamp = float(intconf['ical_dm_amplitude'])
    print("Spatial frequency range for calibration:", kr)

    initial_dm_shape = AOSystem.get_dm_data()
    
    # Example of how to use the DM to create calibration spots
    # and visualize them in the non-blocking viewer
    print("Starting DM calibration sequence...")
    avg_amplitudes = []
    for k in kr:
        print(f"Testing spatial frequency k = {k}")
        data_speck = 0
        additionmapx = dm.make_speckle_kxy(k, 0, DMamp, 0)
        additionmapy = dm.make_speckle_kxy(0, k, DMamp, 0)
        additionmap = additionmapx + additionmapy

        if AOSystem._closed:
            additionmap = AOSystem.convert_voltage_to_cog(additionmap)
        
        AOSystem.set_dm_data(initial_dm_shape + additionmap)
        data_speck_raw = Camera.take_image()
        data_speck = sf.equalize_image(data_speck_raw, **bgds)
        
        cleaned = data_speck - data_nospeck
        #get the spot locations
        guesses = []
        centroids = []
        amplitudes = []
        for (kx, ky) in [(k, 0), (-k, 0), (0, k), (0, -k)]:
            print(kx, ky)
            spotguess = dm.convert_kvecs_pixels(kx, ky, **imageparams)
            guesses.append(spotguess)
            centroids.append(sn.get_spot_centroid(cleaned, guess_spot=spotguess))
            amplitudes.append(sn.get_spot_amplitude(cleaned, guess_spot=spotguess)) 
        

        avg_amplitudes.append(np.mean(amplitudes))

        viewer.set_user_points(centroids)
        viewer.update_data((np.abs(cleaned+1)))
        # Process events to keep UI responsive
        app.processEvents()
        
        # Add a small delay to make the updates visible
        time.sleep(0.1)

    # Restore DM Shape
    AOSystem.set_dm_data(initial_dm_shape)

    popt = sn.curve_fit_parabola(kr, avg_amplitudes)

    # NOTE making a list here for saving compatibility
    popt = [x for x in popt] 
    settings['DM_REGISTRATION']['INTENSITY_CAL']['parabola_parameters'] = popt
    
    # update the calibrated parameters
    with open(config_file, 'wb') as configfile:
        settings.write(configfile)
    
    
    print("Calibration sequence complete.")
    print("Viewer remains open. Close the window when finished.")
    # Return the viewer so we can access it later
    return viewer

if __name__ == "__main__":
    # camera = "Sim"
    # aosystem = "Sim"
    camera = hw.NIRC2Alias()
    aosystem = hw.ClosedAOSystemAlias()

    viewer = run(camera, aosystem, config='fpwfsc/san/sn_config.ini', configspec='fpwfsc/san/sn_config.spec')
    
    # The program will continue here even while the viewer is still open
    print("Main program continues execution while viewer is open")
    
    # Start the Qt event loop and make it exit when the viewer is closed
    app = QApplication.instance()
    
    # This will block until the viewer is closed
    app.exec_()
    
    # After viewer is closed, we can access the points the user selected
    selected_points = [(x, y) for x, y, _ in viewer.selected_points]
    print("User selected points:", selected_points)