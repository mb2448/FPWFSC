import numpy as np
import sys
import threading
from collections import deque
import matplotlib.pyplot as plt

import dm

import qt_clickpoints

sys.path.insert(0, '../')
from common import support_functions as sf
from common import fake_hardware as fhw

def compute_angle(x, y):
    """
    Compute the angle of a point (x, y) with respect to the positive x-axis.
    
    Parameters:
    -----------
    x : float
        x-coordinate of the point
    y : float
        y-coordinate of the point
        
    Returns:
    --------
    angle : float
        Angle in degrees, measured counterclockwise from the positive x-axis.
        Range is [-180, 180] degrees.
    """
    # Use arctan2 to compute the angle
    # arctan2(y, x) returns the angle in radians in the range [-π, π]
    angle_rad = np.arctan2(y, x)
    
    # Convert to degrees
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg


def run(camera=None, aosystem=None, config=None, configspec=None,
        my_deque=None, my_event=None, plotter=None):
    if my_deque is None:
        my_deque = deque()

    if my_event is None:
        my_event = threading.Event()
    
    settings = sf.validate_config(config, configspec)
    #----------------------------------------------------------------------
    # Simulation parameters
    #----------------------------------------------------------------------
    #CAMERA, AO, CORONAGRAPH SETTINGS IN CONFIG FILE
    #----------------------------------------------------------------------
    # Default settings 
    #----------------------------------------------------------------------
    
    #SN Settings
    xcen                = settings['SN_SETTINGS']['xcen']
    ycen                = settings['SN_SETTINGS']['ycen']
    cropsize            = settings['SN_SETTINGS']['cropsize']   
    #----------------------------------------------------------------------
    # Script settings 
    #----------------------------------------------------------------------
    calspot_kx = settings['DM_REGISTRATION']['calspot_kx']
    calspot_ky = settings['DM_REGISTRATION']['calspot_ky']
    calspot_amp= settings['DM_REGISTRATION']['calspot_amp']
    #----------------------------------------------------------------------
    # Load instruments
    #----------------------------------------------------------------------
    if camera == 'Sim' and aosystem == 'Sim':
        CSM      = fhw.FakeCoronagraphOpticalSystem(**settings['SIMULATION']['OPTICAL_PARAMS'])
        AOSystem = fhw.FakeAODMSystem(OpticalModel=CSM, **settings['SIMULATION']['AO_PARAMS'])
        Camera   = fhw.FakeDetector(opticalsystem=CSM,**settings['SIMULATION']['CAMERA_PARAMS'])

    data_nospeck_raw = Camera.take_image()
    data_nospeck = sf.reduce_images(data_nospeck_raw, xcen=xcen, ycen=ycen,
                                              npix=cropsize)

    #plt.imshow(data_nospeck, origin='lower' );plt.show()
    
    speck = dm.make_speckle_kxy(calspot_kx, calspot_ky, calspot_amp, 0, N=22, flipy = False, flipx = False)
    AOSystem.set_dm_data(speck)
    data_ph1_raw    = Camera.take_image()
    data_ph1        = sf.reduce_images(data_ph1_raw, xcen=xcen, ycen=ycen,
                                              npix=cropsize)
    #plt.imshow(data_ph1, origin='lower' );plt.show()
    
    speck = dm.make_speckle_kxy(calspot_kx, calspot_ky, calspot_amp, np.pi/2, N=22, flipy = False, flipx = False)
    AOSystem.set_dm_data(speck)
    data_ph2_raw    = Camera.take_image()
    data_ph2        = sf.reduce_images(data_ph2_raw, xcen=xcen, ycen=ycen,
                                              npix=cropsize)
    #plt.imshow(data_ph2, origin='lower' );plt.show()

    cleaned = 0.5*(data_ph1+data_ph2)-data_nospeck
    #plt.imshow(cleaned, origin='lower' );plt.show()
    points = qt_clickpoints.run_viewer(data = np.log(np.abs(cleaned+1)))
    print(points)

    
if __name__ == "__main__":
    camera = "Sim"
    aosystem = "Sim"
    run(camera, aosystem, config='sn_config.ini', configspec='sn_config.spec')

    print("Generating a speckle at (0, 0)") 
    angle = compute_angle(0, 0)