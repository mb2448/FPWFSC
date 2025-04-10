############################## Import Libraries ###############################
import os
import sys
import numpy as np

# from configobj import ConfigObj
#import matplotlib.pyplot as plt
#import astropy.io.fits as pf
#import sn_math as snm
#import sn_preprocessing as pre
#import sn_filehandling as flh
#import sn_processing as pro


# new stuff
import hcipy as hc
import matplotlib as mpl
import matplotlib.pyplot as plt

# For notebook animations
from matplotlib import animation
# from IPython.display import HTML

class FakeCoronagraph:

    def __init__(self):
        # Scalars
        self.D_pup = 1.
        self.nPup = 512
        self.num_actuators_across = 21
        self.actuator_spacing = self.D_pup / self.num_actuators_across
        self.ref_wave = 2e-6
        self.bw = 0.01
        self.n_wave = 5
        self.samp_foc = 4.
        self.rad_foc = 48
        self.charge = 4
        self.lyot_undersize = 0.95
        self.flux_fudge = 1e8/4
        self.noise_floor = 0.5
        self.x_extent = 264
        self.y_extent = 256
        # self.centerx = 154.22549999999998
        # self.centery = 175.745
        self.centery = 154.22549999999998
        self.centerx = 175.745

        # Grids
        self.pupil_grid = hc.make_pupil_grid(dims=self.nPup, diameter=self.D_pup)
        self.focal_grid = hc.make_focal_grid(
            self.samp_foc,
            self.rad_foc,
            pupil_diameter=self.D_pup,
            focal_length=1.,
            reference_wavelength=self.ref_wave
        )

        # Propagators
        self.prop = hc.FraunhoferPropagator(self.pupil_grid, self.focal_grid)
        self.coro = hc.VortexCoronagraph(self.pupil_grid, self.charge, levels=4, scaling_factor=16)

        # Apertures and masks
        self.aperture = hc.evaluate_supersampled(hc.circular_aperture(self.D_pup), self.pupil_grid, 4)
        self.lyot_mask = hc.evaluate_supersampled(hc.circular_aperture(0.95 * self.D_pup), self.pupil_grid, 4)
        self.lyot_stop = hc.Apodizer(self.lyot_mask)

        # Funny objects
        self.aberration = []
        self.dm = []
        self.ttm = []

    def prop_norm(self,tmp):
        fact = np.max(self.focal_grid.x)*self.pupil_grid.dims[0]/np.max(self.pupil_grid.x)/self.focal_grid.dims[0]
        tmp1 = self.prop(tmp)
        tmp2 = fact*tmp1.electric_field
        tmp3 =hc.Wavefront(tmp2,tmp.wavelength)
        return tmp3

    def make_DM(self):
        """Generate a fourier sine and cosine , up to NFourier cycles per aperture.
                Parameters:
                ----------
                NFourier : int
                    Maximum number for cycles per apertures, use an odd number

                --------
                self.fdm: DeformableMirror
                    Fourier deformable mirror (primary) as a DM object
        """

        influence_functions = hc.make_xinetics_influence_functions(self.pupil_grid, self.num_actuators_across,
                                                                   self.actuator_spacing)
        self.dm = hc.DeformableMirror(influence_functions)

    def make_TTM(self):
        self.ttm = hc.optics.TipTiltMirror(self.pupil_grid)

    def set_photons_levels(self,source_flux,detector_noise):
        self.flux_fudge = source_flux
        self.noise_floor = detector_noise

    def set_filter_model(self,ref_wave_in,bw_in,n_wave_in):
        self.ref_wave = ref_wave_in
        self.bw = bw_in
        self.n_wave = n_wave_in

    def make_aberration(self, aberration_ptv_at_ref_wave):
        tip_tilt = hc.make_zernike_basis(3, self.D_pup, self.pupil_grid, starting_mode=2)
        aberration_ptv = aberration_ptv_at_ref_wave * self.ref_wave
        self.aberration = hc.SurfaceAberration(self.pupil_grid, aberration_ptv, self.D_pup, remove_modes=tip_tilt,
                                               exponent=-3)

    def calc_psf(self):
        brodband_psf = 0
        brodband_img_ref = 0
        flux_fudge_aperture = self.aperture
        for pp in range(0, self.n_wave):
            wave_tmp = self.ref_wave * (1 - self.bw / 2 + self.bw / self.n_wave * pp)
            wf = hc.Wavefront(flux_fudge_aperture, wave_tmp)
            img_ref = self.prop_norm(wf).intensity
            brodband_img_ref += img_ref
            wf = hc.Wavefront(flux_fudge_aperture, wave_tmp)
            wf = self.aberration(wf)
            wf = self.dm(wf)
            wf = self.ttm(wf)
            lyot_plane = self.coro.forward(wf)
            post_lyot_mask = self.lyot_stop(lyot_plane)
            img = self.prop_norm(post_lyot_mask).intensity
            brodband_psf += img
            results = {'unocculted_psf': brodband_img_ref,
                       'occulted_psf': brodband_psf}
        return results


    def take_image(self):
        tmp0 = self.calc_psf()
        size_sim = self.focal_grid.coords.dims
        minusx = self.centerx
        plusx = self.x_extent - self.centerx
        minusy = self.centery
        plusy = self.y_extent - self.centery
        xmin_sim = np.int(size_sim[0]/2 - minusx)
        xmax_sim = np.int(size_sim[0]/2 + plusx)
        ymin_sim = np.int(size_sim[1]/2 - minusy)
        ymax_sim = np.int(size_sim[1]/2 + plusy)
        tmp1 = self.flux_fudge*tmp0['occulted_psf'] / np.max(tmp0['unocculted_psf'])
        tmp2 = np.reshape(tmp1,size_sim)[xmin_sim:xmax_sim,ymin_sim:ymax_sim]
        noise_model = np.random.normal(0, self.noise_floor, tmp2.shape)
        tmp3 = tmp2 + noise_model
        return tmp3

    def take_imag_unocculted(self):
        tmp0 = self.calc_psf()
        size_sim = self.focal_grid.coords.dims
        minusx = self.centerx
        plusx = self.x_extent - self.centerx
        minusy = self.centery
        plusy = self.y_extent - self.centery
        xmin_sim = np.int(size_sim[0]/2 - minusx)
        xmax_sim = np.int(size_sim[0]/2 + plusx)
        ymin_sim = np.int(size_sim[1]/2 - minusy)
        ymax_sim = np.int(size_sim[1]/2 + plusy)
        tmp1 = self.flux_fudge*tmp0['unocculted_psf'] / np.max(tmp0['unocculted_psf'])
        tmp2 = np.reshape(tmp1,size_sim)[xmin_sim:xmax_sim,ymin_sim:ymax_sim]
        noise_model = np.random.normal(0, self.noise_floor, tmp2.shape)
        tmp3 = tmp2 + noise_model
        return tmp3

    def get_dm_shape(self):
        return np.transpose(np.reshape(self.dm.actuators,[self.num_actuators_across,self.num_actuators_across]))/self.ref_wave

    def set_dm_shape(self,DM_update):
        DM_update_tmp = np.transpose(DM_update)
        tmp = np.reshape(DM_update_tmp,self.num_actuators_across**2)
        self.dm.actuators = tmp*self.ref_wave

    def set_TTM(self,tt_commands):
        TT_tmp = tt_commands*self.ref_wave/self.samp_foc/2
        self.ttm.actuators = TT_tmp

    def modulator_move(self,tt_commands):
        TT_tmp = tt_commands*self.ref_wave/self.samp_foc/2
        tmp = self.ttm.actuators
        self.ttm.actuators = tmp + TT_tmp
