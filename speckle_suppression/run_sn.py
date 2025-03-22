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
    #CAMERA SETTINGS IN CONFIG FILE
    #AO PARAMETERS  IN CONFIG FILE
    #CSM PARAMETERS  IN CONFIG FILE
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
    SAN = SpeckleAreaNulling(Camera, AOSystem, 
                               initial_probe_amplitude=1,
                               initial_regularization=1,
                               controlregion_iwa = 3,
                               controlregion_owa = 8,
                               xcenter=xcen,
                               ycenter=ycen,
                               Npix_foc=cropsize,
                               lambdaoverD=4)

    #data_raw = Camera.take_image()
    #data_proc = sf.reduce_images(data_raw, xcen=xcen, ycen=ycen,
    #                                      npix=Npix_foc,
    #                                      flipx = flip_x, flipy=flip_y,
    #                                      rotation_angle_deg=rotation_angle_deg)
    #plt.imshow(data_proc, origin='lower')
    #plt.show() 
    imax=[] 
    ks = []
    plt.ion()
    for k in np.arange(3,11, 0.25):
        speck = dm.make_speckle_kxy(k, 0, 50e-9, 0, N=22, flipy = False, flipx = False)
        AOSystem.set_dm_data(speck)
        data_speck = Camera.take_image()
        data_proc = sf.reduce_images(data_speck, xcen=xcen, ycen=ycen,
                                              npix=CSM.Npix_foc)
        plt.imshow(data_proc, origin='lower')
        plt.draw()
        plt.pause(0.1)
        plt.clf()
        ks.append(k)
        imax.append(np.max(data_proc[130,:]))
        #plt.plot(data_proc[130, :], label = str(k))
        #plt.clf()
        #plt.close()
    plt.close()
if __name__ == "__main__":
    #plotter = pf.LivePlotter()
    
    camera = "Sim"
    aosystem = "Sim"
    run(camera, aosystem, config='sn_config.ini', configspec='sn_config.spec')#, plotter=plotter)