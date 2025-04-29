import os
import sys

import numpy as np
import threading
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path

from fpwfsc.common import dm
from fpwfsc.san import qt_clickpoints
from fpwfsc.san import sn_functions as sn

from fpwfsc.common import support_functions as sf
from fpwfsc.common import fake_hardware as fhw

import ipdb

try:
    from fpwfsc.common import bench_hardware as hw
except:
    raise ImportError


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
    # Script settings 
    #----------------------------------------------------------------------
    calspot_kx = settings['DM_REGISTRATION']['calspot_kx']
    calspot_ky = settings['DM_REGISTRATION']['calspot_ky']
    calspot_amp= settings['DM_REGISTRATION']['calspot_amp']
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
        Camera = camera#hw.Camera.instance()
        AOSystem = aosystem#hw.AOSystem.instance()
    
    bgds = sf.setup_bgd_dict(settings['CAMERA_CALIBRATION']['bgddir'])
    
    # NOTE: In closed loop - this returns a cog, so we need a conversion
    # if not AOSystem._closed:
    #     initial_shape = AOSystem.get_dm_data()
    #     modify_existing = False
    # else:
    #     initial_shape = 0
    #     initial_cogs = AOSystem.get_dm_data()
    #     modify_existing = False
    initial_shape = AOSystem.get_dm_data()

    data_nospeck_raw = Camera.take_image()
    data_nospeck = sf.equalize_image(data_nospeck_raw, **bgds)
    
    speck1 = dm.make_speckle_kxy(calspot_kx, calspot_ky, calspot_amp, 0, N=21, flipy = False, flipx = False)

    if AOSystem._closed:
        speck1 = AOSystem.convert_voltage_to_cog(speck1)

    AOSystem.set_dm_data(initial_shape + speck1)
    data_ph1_raw    = Camera.take_image()
    data_ph1        = sf.equalize_image(data_ph1_raw, **bgds)

    speck2 = dm.make_speckle_kxy(calspot_kx, calspot_ky, calspot_amp, np.pi/2, N=21, flipy = False, flipx = False)
    
    if AOSystem._closed:
        speck2 = AOSystem.convert_voltage_to_cog(speck2) 
    
    AOSystem.set_dm_data(initial_shape + speck2)
    data_ph2_raw    = Camera.take_image()
    data_ph2        = sf.equalize_image(data_ph2_raw, **bgds)

    print("Resetting AO system to initial state")    
    AOSystem.set_dm_data(initial_shape)


    cleaned = 0.5*(data_ph1+data_ph2)-data_nospeck
    #plt.imshow(cleaned, origin='lower' );plt.show()
    points = qt_clickpoints.run_viewer(data = cleaned)
    assert len(points) == 2, "You must click two points"
    c1 = np.array(sn.get_spot_centroid(cleaned, guess_spot=points[0]))
    c2 = np.array(sn.get_spot_centroid(cleaned, guess_spot=points[1]))
    print("Centers: ", c1, c2)
    centered_points = qt_clickpoints.run_viewer(data = cleaned, user_points = [c1, c2])
    measured_angle = compute_angle(*(c2-c1))
    measured_center = 0.5*(c1 + c2)
    print("measured angle: ", measured_angle)
    print("measured center:", measured_center)
    #the /2 is because it is really -kx to kx, and -ky to ky
    measured_lambdaoverD = np.linalg.norm(c2-c1)/np.linalg.norm(np.array([calspot_kx, calspot_ky]))/2
    print("Measured lambda/D: ", measured_lambdaoverD, "pixels")
    settings['DM_REGISTRATION']['MEASURED_PARAMS']['lambdaoverd'] = measured_lambdaoverD
    settings['DM_REGISTRATION']['MEASURED_PARAMS']['centerx'] = measured_center[0]
    settings['DM_REGISTRATION']['MEASURED_PARAMS']['centery'] = measured_center[1]
    settings['DM_REGISTRATION']['MEASURED_PARAMS']['angle'] = measured_angle
    print("Saving settings to: ", config_file)
    with open(config_file, 'wb') as configfile:
        settings.write(configfile)

if __name__ == "__main__":
    Camera = hw.NIRC2Alias()
    #AOSystem = hw.AOSystemAlias()
    AOSystem = hw.ClosedAOSystemAlias()
    # Camera = 'Sim'
    # AOSystem = 'Sim'
    folder = "fpwfsc/san/"

    run(camera=Camera, aosystem=AOSystem, config=folder+'sn_config.ini', configspec=folder+'sn_config.spec')
