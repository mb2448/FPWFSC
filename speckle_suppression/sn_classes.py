import numpy as np
from common import support_functions as sf
import sn_functions as snf
import ipdb
import dm

# TODO: Delete import, used for debugging for now
import matplotlib.pyplot as plt

class SpeckleAreaNulling:
    def __init__(self, camera=None, aosystem=None, initial_probe_amplitude=None, initial_regularization=None,
                 controlregion_iwa=None, controlregion_owa=None, 
                 xcenter=None, ycenter=None, Npix_foc=None, lambdaoverD=None):
        
        self.camera = camera
        self.aosystem = aosystem
        self.initial_probe_amplitude = initial_probe_amplitude
        self.regularization = initial_regularization
        self.probe_amplitude = initial_probe_amplitude
        
        # NOTE: Same dimensions as image
        # NOTE: Referenced to same origin as image

        self.xcenter = xcenter
        self.ycenter = ycenter
        self.lambdaoverD = lambdaoverD
        
        self.controlregion_iwa = controlregion_iwa
        self.controlregion_owa = controlregion_owa
        self.controlregion_iwa_pix = self.controlregion_iwa* \
                                     self.lambdaoverD
        self.controlregion_owa_pix = self.controlregion_owa*\
                                    self.lambdaoverD
        self.imparams = {
                'npix': Npix_foc,
                'xcen': self.xcenter,
                'ycen': self.ycenter,
        }

        # Take reference image
        self.rawI0 = self.camera.take_image()
        
        # reduce the image first

        self.controlregion = snf.create_annular_wedge(self.rawI0, 
                                              self.imparams['xcen'], 
                                              self.imparams['ycen'], 
                                              self.controlregion_iwa_pix, 
                                              self.controlregion_owa_pix, 
                                              -90, 90)
        pix_x = np.arange(self.rawI0.shape[0])
        self.pix_x, self.pix_y = np.meshgrid(pix_x, pix_x)
        #create_annular_wedge(image, xcen, ycen, rad1, rad2, theta1, theta2)

        self.I0 = sf.reduce_images(self.rawI0, **self.imparams)

        # Construct the probes from the wedge
        self.cosines = dm.make_speckle_xy(xs=self.pix_x[self.controlregion==1],
                                      ys=self.pix_y[self.controlregion==1],
                                      amps=1e-1,
                                      phases=0,
                                      centerx=self.imparams['xcen'],
                                      centery=self.imparams['ycen'],
                                      angle=0,
                                      lambdaoverd=self.lambdaoverD,
                                      N=self.aosystem.num_actuators_across)

        self.sines = dm.make_speckle_xy(xs=self.pix_x[self.controlregion==1],
                                      ys=self.pix_y[self.controlregion==1],
                                      amps=1e-1,
                                      phases=np.pi / 2,
                                      centerx=self.imparams['xcen'],
                                      centery=self.imparams['ycen'],
                                      angle=0,
                                      lambdaoverd=self.lambdaoverD,
                                      N=self.aosystem.num_actuators_across)
        
        self.cos_probe = np.sum(self.cosines, axis=-1)
        self.sin_probe = np.sum(self.sines, axis=-1)

        # scale to unity, call probe amplitude in measure
        self.sin_probe /= self.sin_probe.max()
        self.cos_probe /= self.cos_probe.max()

        self.sines /=  self.sin_probe.max()
        self.cosines /=  self.cos_probe.max()

    def _measure(self):
        
        # Take a reference image
        I0 = sf.equalize_image(self.camera.take_image())
        
        # Take the cosine probe images
        # NOTE: Make sure that the I1p / I1m don't look exactly the same
        # NOTE: Check Oya et al. to make sure you are setting the correct probe
        self.images = []
        for probe in [-self.sin_probe, self.sin_probe, -self.cos_probe, self.cos_probe]:
            probe *= self.probe_amplitude
            self.aosystem.set_dm_data(probe, modify_existing="True")
            self.images.append(sf.equalize_image(self.camera.take_image()))
            self.aosystem.set_dm_data(probe, modify_existing="True")


        # unpack the images 
        Im1 = self.images[-4]  # minus sin probe
        Ip1 = self.images[-3]  # plus sin probe
        Im2 = self.images[-2]  # minus cos probe
        Ip2 = self.images[-1]  # plus cos probe
        
        self.I1p = Ip1
        self.I2p = Ip2
        self.I1m = Im1
        self.I2m = Im2 

        # filter by control region
        I0 = I0[self.controlregion==1]
        Ip1 = Ip1[self.controlregion==1]
        Im1 = Im1[self.controlregion==1]
        Ip2 = Ip2[self.controlregion==1]
        Im2 = Im2[self.controlregion==1]

        # Compute the relevant quantities
        dE1 = (Ip1 - Im1) / 4
        dE2 = (Ip2 - Im2) / 4
        dE1sq = (Ip1 + Im1 - 2*I0) / 2
        dE2sq = (Ip2 + Im2 - 2*I0) / 2

        # Regularized sin / cosine coefficients
        sin_coeffs = dE1 / (dE1sq + self.regularization)
        cos_coeffs = dE2 / (dE2sq + self.regularization)

        return sin_coeffs, cos_coeffs

    def _update_probe_amplitude(self, probe_amplitude):
        self.cos_probe /= self.cos_probe.max()
        self.sin_probe /= self.sin_probe.max()
        self.cos_probe *= probe_amplitude
        self.sin_probe *= probe_amplitude        

    def iterate(self, probe_amplitude=None, regularization=None):
        
        if probe_amplitude is not None:
            self._update_probe_amplitude(probe_amplitude)
        
        if regularization is not None:
            self.regularization = regularization

        p, q = self._measure()
        self.pindh = self.controlregion.copy()
        self.qindh = self.controlregion.copy()
        self.pindh[self.controlregion==1] = p
        self.qindh[self.controlregion==1] = q
        p = p[None, None]
        q = q[None, None]

        # bypass and try to plot p/q
        control_sines = p * self.sines
        control_cosines = q * self.cosines

        control_surface = -1 * (np.sum(control_sines, axis=-1) + \
                                np.sum(control_cosines, axis=-1))

        # apply to the deformable mirror
        # NOTE: We are computing 75 meters here
        self.aosystem.set_dm_data(control_surface, modify_existing=True)

        # take image after correction
        I = self.camera.take_image()
        
        return I