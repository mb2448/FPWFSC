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
from matplotlib.colors import LogNorm
from matplotlib.patches import Wedge

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

    SAN = SpeckleAreaNulling(Camera, AOSystem, 
                               initial_probe_amplitude=2.2e-6 / 10,
                               initial_regularization=5e3,
                               controlregion_iwa = 3,
                               controlregion_owa = 8,
                               xcenter=xcen,
                               ycenter=ycen,
                               Npix_foc=cropsize,
                               lambdaoverD=4)

    imax=[] 
    ks = []
    MAX_ITERS = 10
    plt.ion()
    plt.figure(figsize=[12, 3])
    for k in np.arange(MAX_ITERS):


        I_intermediate = SAN.iterate(regularization=5e2)
        
        plt.subplot(131)
        plt.title(f"Probe Amplitude = {SAN.probe_amplitude}")
        plt.imshow(I_intermediate, origin="lower", norm=LogNorm(vmin=1e-2, vmax=1e3), cmap="inferno")
        plt.xlim([275, 375])
        plt.ylim([375, 475])
        plt.colorbar()

        # set up dark hole patch
        ax = plt.gca()
        dh_region = Wedge([xcen, ycen], r=SAN.controlregion_owa_pix, width=SAN.controlregion_owa_pix - SAN.controlregion_iwa_pix,
                          theta1=-90, theta2=90, facecolor="None", edgecolor="w")
        ax.add_patch(dh_region)
        plt.subplot(132)
        plt.imshow(SAN.sin_coeffs_init,  origin="lower", cmap='RdBu_r')
        plt.colorbar()
        plt.xlim([275, 375])
        plt.ylim([375, 475])
        plt.subplot(133)
        plt.imshow(SAN.control_surface, origin="lower", cmap="RdBu_r")
        plt.colorbar()
        # plt.xlim([300, 400])
        # plt.ylim([375, 475])
        plt.draw()
        plt.pause(0.5)
        plt.clf()


        # plt.subplot(121)
        # plt.imshow(I_intermediate, origin="lower", vmin=vmin, vmax=vmax)
        # plt.xlim([200, 500])
        # plt.ylim([200, 500])
        # plt.colorbar()
        # plt.subplot(122)
        # plt.imshow(AOSystem.phase_DM.shaped, cmap="RdBu_r")
        # plt.colorbar()
        # # plt.draw()
        # plt.pause(0.1)
        # plt.clf()
    plt.close()
    # for k in np.arange(3, 11, 0.25):
    #     speck = dm.make_speckle_kxy(k, 0, 20e-9, 0, N=22, flipy = False, flipx = False)
    #     AOSystem.set_dm_data(speck)
    #     data_speck = Camera.take_image()
    #     data_proc = sf.equalize_image(data_speck)
    #     plt.imshow(data_proc, origin='lower', vmin=0, vmax=1e3)
    #     plt.draw()
    #     plt.pause(0.1)
    #     plt.clf()
    #     ks.append(k)
    #     imax.append(np.max(data_proc[130,:]))
    #     #plt.plot(data_proc[130, :], label = str(k))
    #     #plt.clf()
    #     #plt.close()
    #     plt.xlim([200, 600])
    #     plt.ylim([200, 600])
    # plt.close()

if __name__ == "__main__":
    #plotter = pf.LivePlotter()
    
    camera = "Sim"
    aosystem = "Sim"
    run(camera, aosystem, config='sn_config.ini', configspec='sn_config.spec')#, plotter=plotter)