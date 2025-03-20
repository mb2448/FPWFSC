#!/usr/bin/env python
import sys
import numpy as np
import hcipy
from configobj import ConfigObj
import matplotlib.pyplot as plt
import ipdb

sys.path.insert(0, '../')
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
                                  flux = 1e6,
                                  exptime=1,
                                  dark_current_rate=0,
                                  read_noise=1,
                                  flat_field=0,
                                  bias_offset=1000,
                                  include_photon_noise=True,
                                  xsize=1024,
                                  ysize=1024,
                                  field_center_x=333,
                                  field_center_y=433)
    image = Camera.take_image()
    plt.imshow(image);plt.show()
