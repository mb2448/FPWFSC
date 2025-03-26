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
    settings_copy = sf.validate_config(config, configspec)
    #----------------------------------------------------------------------
    # Simulation parameters
    #----------------------------------------------------------------------
    #CAMERA, AO, CORONAGRAPH SETTINGS IN CONFIG FILE
    #SN Settings
    xcen                = settings['SN_SETTINGS']['xcen']
    ycen                = settings['SN_SETTINGS']['ycen']
    cropsize            = settings['SN_SETTINGS']['cropsize']
    settings['SIMULATION']['AO_PARAMS']['initial_rms_wfe'] = 50e-9 * 2 * np.pi / 2.2e-6

    # making a small addition to support normalized intensity calculations
    settings_copy['SIMULATION']['OPTICAL_PARAMS']['INCLUDE_FPM'] = False
    settings_copy['SIMULATION']['AO_PARAMS']['initial_rms_wfe'] = 50e-9 * 2 * np.pi / 2.2e-6


    #----------------------------------------------------------------------
    # Load instruments
    #----------------------------------------------------------------------
    if camera == 'Sim' and aosystem == 'Sim':



        CSM      = fhw.FakeCoronagraphOpticalSystem(**settings['SIMULATION']['OPTICAL_PARAMS'])

        # set up mode basis

        mode_basis = sf.generate_basis_modes(chosen_mode_basis="fourier",
                                                Nmodes=300,
                                                grid_diameter=CSM.Pupil.pupil_diameter,
                                                pupil_grid=CSM.Pupil.pupil_grid)
        mode_basis = sf.orthonormalize_mode_basis(mode_basis,
                                                  CSM.Pupil.aperture)

        settings['SIMULATION']['AO_PARAMS']['modebasis'] = mode_basis
        # settings['SIMULATION']['CAMERA_PARAMS']['aded']

        AOSystem = fhw.FakeAODMSystem(OpticalModel=CSM, **settings['SIMULATION']['AO_PARAMS'])
        Camera   = fhw.FakeDetector(opticalsystem=CSM,**settings['SIMULATION']['CAMERA_PARAMS'])

        CSM_noc  = fhw.FakeCoronagraphOpticalSystem(**settings_copy['SIMULATION']['OPTICAL_PARAMS'])
        Cam_noc  = fhw.FakeDetector(opticalsystem=CSM_noc, **settings_copy['SIMULATION']['CAMERA_PARAMS'])

        # get a contrast value
        image_nofpm = sf.equalize_image(Cam_noc.take_image())
        contrast_norm = image_nofpm.max()
    
    else:
        raise ValueError("Sim only now")

    SAN = SpeckleAreaNulling(Camera, AOSystem, 
                               initial_probe_amplitude=2.2e-6 / 20,
                               initial_regularization=5e-1,
                               controlregion_iwa = 3,
                               controlregion_owa = 8,
                               xcenter=xcen,
                               ycenter=ycen,
                               Npix_foc=cropsize,
                               lambdaoverD=4,
                               contrast_norm=contrast_norm)

    imax=[] 
    ks = []
    MAX_ITERS = 100
    plt.ion()
    plt.figure(figsize=[15, 3])
    mean_ni = []
    iterations = []
    for k in np.arange(MAX_ITERS):

        for i in range(3):

            I_after = SAN.iterate()

            # plot the +sin probed image
            if i == 0:
                I_intermediate = SAN.I1p - SAN.I1m
                coeffs = SAN.sin_coeffs_init
                title = "Sin probe"
                norm = None
                vlim = np.abs(I_intermediate).max()

            # plot the +cos probed image
            elif i == 1:
                I_intermediate = SAN.I2p - SAN.I2m
                coeffs = SAN.cos_coeffs_init
                title = "Cos probe"
                norm = None
                vlim = np.abs(I_intermediate).max()

            else:
                I_intermediate = I_after
                title = "After Correction "
                mean_ni.append(np.median(I_intermediate[SAN.controlregion]))
                iterations.append(k)
                norm = LogNorm(vmin=1e-4, vmax=1)

            plt.subplot(141)
            plt.title(title+"image")
            if norm is None:
                plt.imshow(I_intermediate, origin="lower", vmin=-vlim, vmax=vlim, cmap="coolwarm")
            else:
                plt.imshow(I_intermediate, origin="lower", norm=norm, cmap="inferno")
            plt.xlim([275, 375])
            plt.ylim([375, 475])
            plt.colorbar(label="Normalized Intensity")

            # set up dark hole patch
            ax = plt.gca()
            dh_region = Wedge([xcen, ycen], r=SAN.controlregion_owa_pix, width=SAN.controlregion_owa_pix - SAN.controlregion_iwa_pix,
                            theta1=-90, theta2=90, facecolor="None", edgecolor="w")
            ax.add_patch(dh_region)
            plt.xticks([],[])
            plt.yticks([],[])
            plt.subplot(142)
            plt.title(title+" coefficients")
            plt.imshow(coeffs,  origin="lower", cmap='RdBu_r')
            plt.colorbar()
            plt.xlim([275, 375])
            plt.ylim([375, 475])
            plt.xticks([],[])
            plt.yticks([],[])
            plt.subplot(143)
            plt.title("Total WFE")
            plt.imshow(AOSystem.initial_phase_error.shaped - AOSystem.phase_DM.shaped,
                       origin="lower", cmap="RdBu_r")
            plt.colorbar()
            # plt.xlim([300, 400])
            # plt.ylim([375, 475])
            plt.xticks([],[])
            plt.yticks([],[])
            plt.subplot(144)
            plt.plot(iterations, mean_ni)
            plt.ylabel("Median NI")
            plt.xlabel("SAN Iterations")
            plt.yscale("log")
            plt.xticks([],[])
            plt.yticks([],[])
            plt.draw()
            plt.pause(0.5)
            plt.clf()
            ipdb.set_trace()

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