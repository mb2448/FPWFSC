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
    calspot_k = settings['QACITS_SPECKLE']['calspot_k'] 
    calspot_amp= settings['DM_REGISTRATION']['calspot_amp']
    calspot_k = float(calspot_k)
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
    
    speck1 = dm.make_speckle_kxy(calspot_k, calspot_k, calspot_amp, 0, N=60, flipy = False, flipx = False)
    speck2 = dm.make_speckle_kxy(calspot_k, -calspot_k, calspot_amp, 0, N=60, flipy = False, flipx = False)

    if AOSystem._closed:
        speck = AOSystem.convert_voltage_to_cog(speck1 + speck2)

    AOSystem.set_dm_data(initial_shape + speck)
    img = sf.equalize_image(Camera.take_image(), **bgds)

if __name__ == "__main__":
    Camera = hw.NIRC2Alias()
    #AOSystem = hw.AOSystemAlias()
    AOSystem = hw.ClosedAOSystemAlias()
    # Camera = 'Sim'
    # AOSystem = 'Sim'
    folder = "/home/mcisse/FPWFSC/fpwfsc/san/"

    run(camera=Camera, aosystem=AOSystem, config=folder+'sn_config.ini', configspec=folder+'sn_config.spec')
