import numpy as np
import sys
import ipdb
from ..common import support_functions as sf
from . import sn_functions as snf
from . import dm

# TODO: Delete import, used for debugging for now
import matplotlib.pyplot as plt

class SpeckleAreaNulling:
    def __init__(self, camera=None, aosystem=None, initial_probe_amplitude=None, initial_regularization=None,
                 controlregion_iwa=None, controlregion_owa=None,
                 xcenter=None, ycenter=None, Npix_foc=None, lambdaoverD=None,
                 contrast_norm=1, flipx=False, flipy=False):

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
        self.CN = contrast_norm
        self.flipx = flipx
        self.flipy = flipy

        # Take reference image
        self.rawI0 = sf.equalize_image(self.camera.take_image()) / self.CN

        # reduce the image first
        self.controlregion = snf.create_annular_wedge(self.rawI0,
                                              self.imparams['xcen'],
                                              self.imparams['ycen'],
                                              self.controlregion_iwa_pix,
                                              self.controlregion_owa_pix,
                                              -90, 90)
        pix_x = np.arange(self.rawI0.shape[0])
        self.pix_x, self.pix_y = np.meshgrid(pix_x, pix_x)

        self.I0 = sf.reduce_images(self.rawI0, **self.imparams)
        self.sines = []
        self.cosines = []

        # grab indices of control region
        control_indices = np.where(self.controlregion) # where true
        for yi, xi in zip(*control_indices):

            # Construct the probes from the wedge
            cos = dm.make_speckle_xy(xs=xi,
                                        ys=yi,
                                        amps=1,
                                        phases=0,
                                        centerx=self.imparams['xcen'],
                                        centery=self.imparams['ycen'],
                                        angle=0,
                                        lambdaoverd=self.lambdaoverD,
                                        N=self.aosystem.num_actuators_across,
                                        which="cos",
                                        flipy=self.flipy,
                                        flipx=self.flipx)

            sin = dm.make_speckle_xy(xs=xi,
                                        ys=yi,
                                        amps=1,
                                        phases=0,
                                        centerx=self.imparams['xcen'],
                                        centery=self.imparams['ycen'],
                                        angle=0,
                                        lambdaoverd=self.lambdaoverD,
                                        N=self.aosystem.num_actuators_across,
                                        which="sin",
                                        flipy=self.flipy,
                                        flipx=self.flipx)

            self.sines.append(sin)
            self.cosines.append(cos)
        self.cos_probe = np.sum(self.cosines, axis=0)
        self.sin_probe = np.sum(self.sines, axis=0)

        # scale to unity, call probe amplitude in measure
        self.sin_probe /= self.sin_probe.max()
        self.cos_probe /= self.cos_probe.max()

        self.cos_probe_init = self.cos_probe.copy()
        self.sin_probe_init = self.sin_probe.copy()

        self.sines /=  self.sin_probe.max()
        self.cosines /=  self.cos_probe.max()

    def _measure(self, probe_amplitude=None):
        """hidden method to take the series of probe measurements

        Parameters
        ----------
        probe_amplitude : float, optional
            scalar amplitude to apply to the sine and cosine probes,
            by default None, which uses the initial probe specified.

        Returns
        -------
        sin_coeffs, cos_coeffs
            coefficients determined for the correction
        """

        if probe_amplitude is not None:
            self.probe_amplitude = probe_amplitude

        # Take a reference image
        I0 = sf.equalize_image(self.camera.take_image()) / self.CN

        # Take the cosine probe images
        # NOTE: Make sure that the I1p / I1m don't look exactly the same
        # NOTE: Check Oya et al. to make sure you are setting the correct probe
        self.images = []
        for probe in [-self.sin_probe, self.sin_probe, -self.cos_probe, self.cos_probe]:
            probe_scaled = probe * self.probe_amplitude
            self.aosystem.set_dm_data(probe_scaled)
            self.images.append(sf.equalize_image(self.camera.take_image()) / self.CN)
            self.aosystem.set_dm_data(-probe_scaled)

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

        # +sin probe
        Ip1 = Ip1[self.controlregion==1]

        # -sin probe
        Im1 = Im1[self.controlregion==1]

        # +cos probe
        Ip2 = Ip2[self.controlregion==1]

        # -cos probe
        Im2 = Im2[self.controlregion==1]

        # Compute the relevant quantities
        dE1 = (Ip1 - Im1) / 4
        dE2 = (Ip2 - Im2) / 4
        dE1sq = (Ip1 + Im1 - 2*I0) / 2
        self.dE1sq = self.controlregion.copy().astype(np.float64)
        dE2sq = (Ip2 + Im2 - 2*I0) / 2
        self.dE2sq = self.controlregion.copy().astype(np.float64)
        self.dE1sq[self.controlregion==1] = dE1sq
        self.dE2sq[self.controlregion==1] = dE2sq

        # Filter the quantities dE1 and dE2 below the regularization threshold

        # # Regularized sin / cosine coefficients
        sin_coeffs = dE1 / (dE1sq + self.regularization)
        cos_coeffs = dE2 / (dE2sq + self.regularization)
        # # sin_coeffs = dE1 / (dE1sq)
        # # cos_coeffs = dE2 / (dE2sq)
        # sin_coeffs[np.abs(sin_coeffs) > self.regularization] = 0
        # cos_coeffs[np.abs(cos_coeffs) > self.regularization] = 0

        self.sin_coeffs_init = self.controlregion.copy().astype(np.float64)
        self.cos_coeffs_init = self.controlregion.copy().astype(np.float64)

        self.sin_coeffs_init[self.controlregion==1] = sin_coeffs
        self.cos_coeffs_init[self.controlregion==1] = cos_coeffs

        return sin_coeffs, cos_coeffs

    def iterate(self, probe_amplitude=None, regularization=None):
        """Advance the SAN algorithm one iteration

        Parameters
        ----------
        probe_amplitude : float, optional
            scalar amplitude to apply to the sine and cosine probes,
            by default None, which uses the initial probe specified.
        regularization : float, optional
            The intensity threshold below which corrections are not computed,
            by default None

        Returns
        -------
        ndarray
            array containing the image after the SAN algorithm has applied a correction
        """

        if regularization is not None:
            self.regularization = regularization

        p, q = self._measure(probe_amplitude=probe_amplitude)

        # Broadcast the shapes of p and q to allow element-wise multiplication
        p = p[..., None, None]
        q = q[..., None, None]

        # bypass and try to plot p/q
        control = self.probe_amplitude * (p * self.sines + q * self.cosines)

        # self.sin_coeffs_init = np.sum(control_sines, axis=-1)
        # self.cos_coeffs_init = np.sum(control_cosines, axis=-1)

        control_surface = -1* np.sum(control, axis=0)
        control_surface *= 0.1 / control_surface.max()

        # control_surface *= self.aosystem.OpticalModel.wavelength / (2 * np.pi)
        # apply to the deformable mirror
        self.control_surface = control_surface
        self.aosystem.set_dm_data(control_surface)

        # take image after correction
        I = sf.equalize_image(self.camera.take_image()) / self.CN

        return I
