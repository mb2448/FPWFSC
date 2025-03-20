#!/usr/bin/env python
import sys
import numpy as np
import hcipy
from configobj import ConfigObj
import matplotlib.pyplot as plt

from common import plotting_funcs as pf
from common import classes as ff_c
from common import fake_hardware as fhw
from common import support_functions as sf


if __name__ == "__main__":

    TelescopeAperture = ff_c.Aperture(Npix_pup=128,
                             aperturename='keck',
                             rotation_angle_aperture=0)
    Lyotcoronagraph = ff_c.LyotCoronagraph(Npix_foc=128, IWA_mas=150, mas_pix=10, pupil_grid=TelescopeAperture.pupil_grid)
    LyotStop          = ff_c.Aperture(Npix_pup=128,
                                      rotation_angle_aperture=0,
                                      aperturename='NIRC2_incircle_mask')

    TelescopeAperture.display()
    Lyotcoronagraph.display()
    LyotStop.display()

    CSM = ff_c.CoronagraphSystemModel(telescopeaperture=TelescopeAperture,
                       coronagraph=Lyotcoronagraph,
                       lyotaperture=LyotStop,
                       Npix_foc=128,
                       mas_pix=10,
                       wavelength=2.2e-6)
    Camera = fhw.FakeDetector(opticalsystem=CSM,
                                  flux = 1e5,
                                  exptime=1,
                                  dark_current_rate=0,
                                  read_noise=10,
                                  flat_field=0,
                                  include_photon_noise=True,
                                  xsize=1024,
                                  ysize=1024,
                                  field_center_x=333,
                                  field_center_y=433)

    AOSystem = fhw.FakeAODMSystem(OpticalModel=CSM,
                       modebasis=None,
                       initial_rms_wfe=0,
                       rotation_angle_dm = 0,
                       num_actuators_across=22,
                       actuator_spacing=None,
                       seed=None)

    # set DM surface to a sinusoid
    x = np.linspace(-1, 1, 22)
    x, y = np.meshgrid(x, x)

    sin_err_acts = np.sin(x * 4) / 10
    AOSystem.set_dm_data(sin_err_acts.ravel())

    image = Camera.take_image(focal_wf = CSM.generate_psf_efield())
    plt.imshow(image);plt.show()
