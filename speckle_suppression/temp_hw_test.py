#!/usr/bin/env python
import sys
import threading
import numpy as np
from collections import deque
import hcipy
from configobj import ConfigObj
import time
import matplotlib.pyplot as plt
import dm
import ipdb

sys.path.insert(0, '../')
from common import plotting_funcs as pf
from common import classes as ff_c
from common import fake_hardware as fhw
from common import support_functions as sf

from sn_classes import SpeckleAreaNulling

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
    #SN Settings
    xcen                = settings['SN_SETTINGS']['xcen']
    ycen                = settings['SN_SETTINGS']['ycen']
    cropsize            = settings['SN_SETTINGS']['cropsize']   
    #----------------------------------------------------------------------
    # Load instruments
    #----------------------------------------------------------------------
    if camera == 'Sim' and aosystem == 'Sim':
        CSM      = fhw.FakeCoronagraphOpticalSystem(**settings['SIMULATION']['OPTICAL_PARAMS'])
        AOSystem = fhw.FakeAODMSystem(OpticalModel=CSM, **settings['SIMULATION']['AO_PARAMS'])
        Camera   = fhw.FakeDetector(opticalsystem=CSM,**settings['SIMULATION']['CAMERA_PARAMS'])
    
    else:
        raise ValueError("Sim only now")

    return CSM, AOSystem, Camera

if __name__ == "__main__":
    camera = "Sim"
    aosystem = "Sim"
    CSM, AOSystem, Camera = run(camera, aosystem, config='sn_config.ini', configspec='sn_config.spec')
    print("Validating no mystery rotations in the Camera")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(Camera.take_image(), origin='lower')
    axes[0].set_title('Camera Image')
    axes[1].imshow(np.array(CSM.focal_efield.power.shaped), origin='lower')
    axes[1].set_title('Focal Plane Electric Field Power')
    plt.tight_layout()
    plt.show()

    print("Showing Poisson noise, note array indexing in the x and y are flipped")
    for i in range(10):
        a = Camera.take_image()
        print(a[423, 283])