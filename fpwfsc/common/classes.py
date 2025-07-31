#References below
#Keller2012
#Keller, Christoph U., et al. "Extremely fast focal-plane wavefront sensing for extreme adaptive optics." Adaptive Optics Systems III. Vol. 8447. SPIE, 2012.
#
#Korkiakoski14
#Korkiakoski, Visa, et al. "Fast & Furious focal-plane wavefront sensing." Applied optics 53.20 (2014): 4565-4579.
#
#Wilby2018
#Wilby, Michael J., et al. "Laboratory verification of Fast & Furious phase diversity: Towards controlling the low wind effect in the SPHERE instrument." Astronomy & Astrophysics 615 (2018): A34.
#
#Bos2021
#Bos, Steven P., et al. "Fast and furious focal-plane wavefront sensing at WM Keck Observatory." Techniques and Instrumentation for Detection of Exoplanets X. Vol. 11823. SPIE, 2021.
import ipdb
import numpy as np
import hcipy
import matplotlib.pyplot as plt

from . import support_functions as sf
from . import vandamstrehl as vDs
from . import make_subaru_aperture as msa
from . import make_keck_aperture

# import support_functions as sf
# import vandamstrehl as vDs
# import make_subaru_aperture as msa
# import make_keck_aperture

class Aperture:
    """A class to generate the aperture of the optical system.  This requires a
    separate function (that must be written) to precisely generate the pupil
    Parameters
    ---------
    Npix_pup : int
        The number of pixels of resolution across the pupil you desire
    diameter : float
        The diameter in meters across the pupil model (must be larger than the pupil itself, eg 12 meters for 10 m Keck telescope)
    aperturename : string
        The name of the aperture
    rotation_angle_aperture: float (deg)
        The rotation angle in degrees of the pupil

    Returns:
    ---------
    aperture : hcipy Field
    """

    def __init__(self, Npix_pup=None,
                 aperturename=None, rotation_angle_aperture=None, rotation_primary_deg=None):
        print("Initializing aperture")
        self.Npix_pup = Npix_pup
        self.aperturename = aperturename
        self.rotation_angle_aperture = rotation_angle_aperture
        self.rotation_primary_deg = rotation_primary_deg
        if aperturename == 'subaru':
            self.diameter = 8.0
            self.pupil_grid = hcipy.make_pupil_grid(Npix_pup, diameter=self.diameter)
            self.aperture, self.pupil_diameter = msa.generate_pupil(n = self.Npix_pup, outer = 1, grid = self.pupil_grid,
                                                                    inner = msa.INNER_RATIO,
                                                                    scale = 1,
                                                                    angle= self.rotation_angle_aperture,
                                                                    oversample = 8,
                                                                    spiders = True,
                                                                    actuators = True)
            self.aperture = self.aperture.reshape(self.Npix_pup**2)
        else:
            #this is the Keck case for now
            self.diameter = 11.3
            self.pupil_grid = hcipy.make_pupil_grid(Npix_pup, diameter=self.diameter)
            self.aperture, self.pupil_diameter  = make_keck_aperture.get_aperture(aperturename=self.aperturename,
                                pupil_grid=self.pupil_grid,
                                rotation_angle_aperture=self.rotation_angle_aperture, 
                                )
            # #fix this !
            # self.pupil_grid = hcipy.make_pupil_grid(Npix_pup, diameter=self.pupil_diameter1)
            # self.aperture, self.pupil_diameter  = make_keck_aperture.get_aperture(aperturename=self.aperturename,
            #                     pupil_grid=self.pupil_grid,
            #                     rotation_angle_aperture=self.rotation_angle_aperture, 
            #                     )
            
        return

    def display(self):
        """plots the aperture"""
        hcipy.imshow_field(self.aperture)
        plt.show()
        return self.aperture


class LyotCoronagraph:

    def __init__(self, Npix_foc=None, IWA_mas=None, mas_pix=None, pupil_grid=None):
        """
        Parameters
        ----------

        Npix_foc : int
            size of focal plane array along side dimension
        IWA_mas : float
            Lyot coronagraph radius in miliarcseconds
        mas_pix : float
            Pixelscale in miliarcseconds
        pupil_grid : hcipy.PupilGrid
            Pupil grid the wavefront is defined on
        wavelengt : float
            The wavelength in meters
        """
        self.Npix_foc = Npix_foc
        self.IWA_mas = IWA_mas
        self.IWA_rad = np.radians(self.IWA_mas/1000./3600.)
        self.mas_pix = mas_pix
        self.rad_pix    = np.radians(self.mas_pix / 1000. / 3600.)
        self.pupil_grid = pupil_grid
        # init field generator
        aperture = hcipy.make_circular_aperture(diameter=self.IWA_rad*2)
        spot_generator = hcipy.make_obstruction(aperture)

        self.focal_grid = hcipy.make_uniform_grid(
             [self.Npix_foc, self.Npix_foc],
             [self.Npix_foc*self.rad_pix, self.Npix_foc*self.rad_pix])
        #self.focal_grid.wavelength = self.wavelength
        #self.pupil_grid.wavelength = self.wavelength
        self.fpm = spot_generator(self.focal_grid)

    def forward_tolyot(self, input_wavefront, include_fpm=True):

        prop = hcipy.FraunhoferPropagator(self.pupil_grid, self.focal_grid)
        focal = prop.forward(input_wavefront)
        if include_fpm:
            focal.electric_field *= self.fpm
        lyot_efield = prop.backward(focal)
        return lyot_efield

    def display(self):
        """plots the aperture"""
        hcipy.imshow_field(self.fpm)
        plt.show()
        return self.fpm

class CoronagraphSystemModel:
    """The optical system model.
    A class to generate the system model
    Parameters
    ---------
    telescopeaperture : classes.aperture (above)
        The telecope system aperture
    coronagraph : XYZ to fill in
    lyotaperture : XYZ to fill in

    Npix_foc : int
        Number of pixels to generate model of psf on (does not need to be the same as the camera image size)
    mas_pix : float
        The number of milliarcseconds per pixel in the model.  Needs to match camera plate scale
    wavelength : float (physical units, in meters (not um/nm)
        The effective central wavelength of the filter in your (real) optical system
    flux : float
        photons/second in psf.  mainly useful for generating fake images.  DOES NOT ADD PHOTON NOISE.
    Returns:
    ---------
    """
    def __init__(self, telescopeaperture=None,
                       coronagraph=None,
                       lyotaperture=None,
                       Npix_foc=None,
                       mas_pix=None,
                       wavelength=None,
                       flipx=None,
                       flipy=None,
                       rotation_angle_deg=None,
                       include_fpm=True):
        print("Initializing system model")

        self.Pupil      = telescopeaperture
        self.FocalSpot  = coronagraph
        self.LyotStop   = lyotaperture
        self.pupil_grid = self.Pupil.pupil_grid
        self.Npix_foc   = Npix_foc
        self.mas_pix    = mas_pix
        self.rad_pix    = np.radians(self.mas_pix / 1000. / 3600.)
        self.wavelength = wavelength #in meters
        self.flipx = flipx
        self.flipy = flipy
        self.rotation_angle_deg = rotation_angle_deg
        self.include_fpm = include_fpm
        self.focal_grid = hcipy.make_uniform_grid(
             [self.Npix_foc, self.Npix_foc],
             [self.Npix_foc*self.rad_pix, self.Npix_foc*self.rad_pix])
        #generate the propagator, inherits from the aperture
        self.propagator = hcipy.FraunhoferPropagator(self.Pupil.pupil_grid,
                                                     self.focal_grid)

        #generate the reference aperture
        self.ref_pupil_field = hcipy.Wavefront(self.Pupil.aperture*1,
                                           wavelength=self.wavelength)
        ##copy by value!
        self.pupil_efield = self.ref_pupil_field.copy()
        self.focal_efield = self.generate_psf_efield().copy()
        return

    def update_pupil_wavefront(self, applied_phase):
        self.pupil_efield = hcipy.Wavefront(self.Pupil.aperture * \
                                                 np.exp(1j * applied_phase),
                                            wavelength=self.wavelength)
        #self.pupil_efield.total_power = 1
        self.focal_efield = self.generate_psf_efield()
        return

    def generate_psf_efield(self):
        """Generate a psf.
        Parameters
        ----------

        Returns
        ----------
        focal_wf
        """
        #XYZ somehwere here put in the flip
        #print("Flip put in here somewhere")
        lyot_wf = self.FocalSpot.forward_tolyot(self.pupil_efield, include_fpm=self.include_fpm)
        lyot_wf.electric_field *= self.LyotStop.aperture
        focal_wf = self.propagator(lyot_wf)
        focal_wf = sf.rotate_and_flip_wavefront(focal_wf,
                                                angle=self.rotation_angle_deg,
                                                flipx=self.flipx,
                                                flipy=self.flipy)
        return focal_wf

class SystemModel:
    """The optical system model.
    A class to generate the system model
    Parameters
    ---------
    aperture : classes.aperture (above)
        The system aperture
    Npix_foc : int
        Number of pixels to generate model of psf on (does not need to be the same as the camera image size)
    mas_pix : float
        The number of milliarcseconds per pixel in the model.  Needs to match camera plate scale
    wavelength : float (physical units, in meters (not um/nm)
        The effective central wavelength of the filter in your (real) optical system
    flux : float
        photons/second in psf.  mainly useful for generating fake images.  DOES NOT ADD PHOTON NOISE.
    Returns:
    ---------
    """
    def __init__(self, aperture=None,
                       Npix_foc=None,
                       mas_pix=None,
                       wavelength=None):
        print("Initializing system model")
        self.Pupil      = aperture
        self.pupil_grid = self.Pupil.pupil_grid
        self.Npix_foc   = Npix_foc
        self.mas_pix    = mas_pix
        self.rad_pix    = np.radians(self.mas_pix / 1000. / 3600.)
        self.wavelength = wavelength
        self.focal_grid = hcipy.make_uniform_grid(
             [self.Npix_foc, self.Npix_foc],
             [self.Npix_foc*self.rad_pix, self.Npix_foc*self.rad_pix])
        #generate the propagator, inherits from the aperture
        self.propagator = hcipy.FraunhoferPropagator(self.Pupil.pupil_grid,
                                                     self.focal_grid)
        #generate the fourier transform operator
        self.fourier_transform = hcipy.make_fourier_transform(self.focal_grid, q=1, fov=1)
        #generate the reference aperture
        self.ref_pupil_field = hcipy.Wavefront(self.Pupil.aperture*1,
                                           wavelength=self.wavelength)
        #compute the FT of the aperture
        self.a= self.propagator(self.ref_pupil_field)
        #compute the reference psf electric field and image
        self.ref_psf_efield = self.a.electric_field
        self.ref_psf   = self.a.power
        #set the current pupil efield to the reference field
        #copy by value!
        self.pupil_efield = self.ref_pupil_field.copy()
        self.focal_efield = self.ref_psf_efield.copy()
        return

    def update_pupil_wavefront(self, applied_phase):
        self.pupil_efield = hcipy.Wavefront(self.Pupil.aperture * \
                                                 np.exp(1j * applied_phase),
                                            wavelength=self.wavelength)
        #self.pupil_efield.total_power = 1
        self.focal_efield = self.generate_psf_efield()
        return

    def generate_psf_efield(self):
        """Generate a psf.
        Parameters
        ----------

        Returns
        ----------
        focal_wf
        """
        #XYZ somehwere here put in the flip
        #print("Flip put in here somewhere")
        focal_wf = self.propagator(self.pupil_efield)
        return focal_wf


class FastandFurious:
    def __init__(self,
                 SystemModel=None,
                 gain=None,
                 leak_factor=None,
                 chosen_mode_basis=None,
                 epsilon=None,
                 number_of_modes=None,
                 control_odd_modes=True,
                 control_even_modes=True,
                 dm_command_boost=None):
        print("Initializing F&F")
        print("Importing optical system model parameters")
        #careful these are passed by reference!
        ## correcting the Fraunhofer propagation for the math of F&F
        ## we have to rotate the electric field by 90 degrees in the complex plane.
        ## this is because the factor of 1/i that is added to Fraunhofer propagator after the Fourier transform.
        self.ref_psf_efield = SystemModel.ref_psf_efield.copy() \
                              * np.exp(1j * np.pi / 2)
        self.ref_psf = SystemModel.ref_psf.copy()

        self.pupildiameter = SystemModel.Pupil.diameter
        self.pupil_grid = SystemModel.Pupil.pupil_grid.copy()
        self.focal_grid = SystemModel.focal_grid.copy()
        self.aperture = SystemModel.Pupil.aperture.copy()
        #aperture limited to reasonable values, if grayscale
        self.corr_aper = self.aperture > 1e-5
        self.wavelength = SystemModel.wavelength
        #Careful--these are passed by reference
        self.propagator = SystemModel.propagator
        self.fourier_transform = SystemModel.fourier_transform

        print("Generating mode basis")
        self.chosen_mode_basis= chosen_mode_basis

        if self.chosen_mode_basis != 'pixel':
            self.mode_basis = sf.generate_basis_modes(chosen_mode_basis=
                                                   self.chosen_mode_basis,
                                                   Nmodes=number_of_modes,
                                                   grid_diameter=self.pupildiameter,
                                                   pupil_grid=self.pupil_grid)
            self.mode_basis = sf.orthonormalize_mode_basis(self.mode_basis,
                                                           self.aperture)
        else:
            self.mode_basis = None

        print("Configuring loop parameters")
        self.gain = gain
        self.leak_factor = leak_factor
        self.control_odd_modes = control_odd_modes
        self.control_even_modes = control_odd_modes
        self.dm_command_boost = dm_command_boost

        #Regularization parameter
        self.epsilon = epsilon*np.max(self.ref_psf.max())

        #iteration parameters
        self.current_image  = None
        self.previous_image = None
        self.strehl_est     = None
        #the phase diversity term, set to zero initially
        print("Initializing diversity phase to zero")
        self.phi_i = np.zeros_like(self.aperture)
        self.phase_DM = np.zeros_like(self.aperture)
        return

    def initialize_first_image(self, data):
        """
        Initializes the input image to the previous image

        Parameters
        ---------
        data - numpy array
            the camera image data, should be same size as the
            model reference psf, and aligned to it

        Outputs
        -------
        None
        """
        self.previous_image = self.process_raw_image(data)
        return

    def initialize_diversity_phase(self, phase):
        """
        Initializes the input phase to the diversity_phase

        Parameters
        ---------
        phase - Hcipy Field
            the diversity phase (may be zero)

        Outputs
        -------
        None
        """
        assert type(phase) is hcipy.Field, "Initial phase must be Field object"
        assert phase.shape == self.aperture.shape, "Initial phase must be same size as pupil aperture"
        self.phi_i = phase
        return

    def iterate(self, raw_data):
        #self.phi_i *= -1
        self.current_image = self.process_raw_image(raw_data)
        self.strehl_est = self.estimate_strehl()
        #calculate odd/even focal plane terms using focal data
        y, v_abs = self.solve_yv(self.previous_image)
        if self.phi_i.any() != 0:
            # If there is phase diversity info we can give a good estimate of the sign
            v_sign = self.sign_v(self.current_image,
                                 self.previous_image,
                                 self.phi_i,
                                 y)
        else:
            #guess the sign from reference psf
            v_sign = np.sign(self.ref_psf_efield)
        #reconstruct the even wavefront
        v = hcipy.Wavefront(v_sign * v_abs.electric_field,
                            wavelength=self.wavelength)
        #get the total wavefront to control
        tot = self.control_even_odd(oddpart=y, evenpart=v,
                         control_odd_modes=self.control_odd_modes,
                         control_even_modes=self.control_even_modes)
        #return the phase estimate.
        phi_FF = self.propagator.backward(tot).imag * self.corr_aper
        if self.chosen_mode_basis != 'pixel':
            modal_coeffs = sf.modal_decomposition(phi_FF,
                                                  self.mode_basis)
            phi_FF = sf.modal_recomposition(modal_coeffs,
                                            self.mode_basis,
                                            self.aperture)
        else:
            phi_FF = sf.remove_piston(phi_FF, aperture=self.corr_aper)
        #multiply by aperture function to prevent numerical issues
        phi_FF *= self.corr_aper

        self.phi_i = phi_FF
        #apply gains
        self.phase_DM *= self.leak_factor
        self.phi_i *= self.gain
        self.phase_DM += self.phi_i.copy()

        #update last image
        self.previous_image = self.current_image.copy()
        self.current_image = None
        return self.phase_DM

    def control_even_odd(self, oddpart=None, evenpart=None,
                         control_odd_modes=True,
                         control_even_modes=True):
        """Reconstructs the full wavefront from odd and even parts
        solved for by F&F algorithm

        Parameters
        __________
        oddpart/evenpart - Field
            The odd/even part of the wavefront
        control_even/odd_modes - Boolean
            control even and odd modes

        Returns
        ----------
        tot - Wavefront
            The total wavefront to control
        """
        # checking if we control the even and odd modes (cast to 1/0)
        even = int(control_even_modes)
        odd = int(control_odd_modes)
        #adding both components together
        tot = hcipy.Wavefront(even * evenpart.electric_field \
                             -odd* 1j * oddpart.electric_field,
                             wavelength=self.wavelength)
        return tot

    def sign_v(self, current_image, last_image, phi_d, odd_est):
        ''' Estimates the sign of the even component.

        Based on the algorithm of Keller2012 and the code of Wilby2018

        Parameters
        ----------
        current_image : Field
            The image of the PSF.
        last_image : Field
            The image of the previous PSF.
        phi_d: Field
            The diversity phase.
        odd_est: Wavefront
            Estimate of the odd electric field
        Returns
        -------
        v_sign : Field
            Estimate of the sign of the even electric field.
        '''
        # first we find the even arts of the data
        p_e_1, _ = sf.fouriersplit(current_image,
                                   self.fourier_transform)
        p_e_2, _ = sf.fouriersplit(last_image,
                                   self.fourier_transform)
        # finding the phase diversity electric field
        p_d = self.propagator(hcipy.Wavefront(
                  self.aperture * phi_d,
                  wavelength=self.wavelength))

        # correcting the Fourier transform for the math of F&F
        p_d.electric_field *= np.exp(1j * np.pi / 2)
        # splitting that into the even and odd parts
        v_d, y_d = sf.fouriersplit(p_d,
                                   self.fourier_transform)
        v_d = hcipy.Wavefront(v_d, wavelength=self.wavelength)
        y_d = hcipy.Wavefront(y_d, wavelength=self.wavelength)
        #finding v
        v = ((p_e_2 - p_e_1 - \
                (v_d.power + y_d.power + \
                 2 * odd_est.electric_field * \
                 y_d.electric_field * current_image.grid.weights)) / \
              (2 * v_d.electric_field * np.sqrt(current_image.grid.weights))) / \
             np.sqrt(current_image.grid.weights)
        #keep only the sign
        v_sign = np.sign(v)
        return v_sign

    def solve_yv(self, proc_data):
        """Estimates the odd component and the absolute value of the even component.

        Based on the algorithm of Keller2012 and the code of Wilby2018 .

        Parameters
        ----------
        proc_data : Field
            The processed frame

        Returns
        -------
        y : Wavefront
            The estimate of the odd part of the electric field.
        v_abs : Wavefront
            The estimate of the absolute value of the even part of the electric field.
        """
        #split processed data to odd/even parts
        p_e, p_o = sf.fouriersplit(proc_data,
                                   self.fourier_transform)
        #Eqn 16 from Korkiakoski2014, solve for odd part
        y = hcipy.Wavefront((self.ref_psf_efield * p_o) /
                            (2 * self.ref_psf + self.epsilon),
                            wavelength=self.wavelength)
        #Solve for even part (note dropping the old strehl term,
        #assuming we use the standard normalization in Kork14)
        v_abs = hcipy.Wavefront(
                    np.sqrt(np.abs(p_e - self.ref_psf - y.power)
                              ) /
                    np.sqrt(proc_data.grid.weights),
                           wavelength=self.wavelength)
        return y, v_abs

    def process_raw_image(self, data):
        """
        Converts to a Field and scales the data to match the reference_psf.
        A simpler implementation with similar performance to Korkiakoskk2014
        Eqn 12-13

        Parameters
        ---------
        data - numpy array
            The raw data from the detector, a 2d array of the right size
            and aligned with the reference psf

        Returns
        --------
        p - hcipy Field
            the raveled data, converted to a field and scaled
        """
        data = hcipy.Field(data.ravel(), self.focal_grid)
        p = data*np.max(self.ref_psf)/np.max(data)
        return p

    def process_raw_image_old(self, data):
        """Korkiakoski2014 Eq 12 - 13.
        Converts to a Field and scales the data to match the reference_psf
        Parameters
        ---------
        data - numpy array
            The raw data from the detector, a 2d array of the right size
            and aligned with the reference psf

        Returns
        --------
        p - hcipy Field
            the raveled data, converted to a field and scaled
        """
        data = hcipy.Field(data.ravel(), self.focal_grid)
        pn = data * np.sum(self.ref_psf) / np.sum(data)
        p = pn + (1 - np.max(pn)/np.max(self.ref_psf))*self.ref_psf
        return p

    def estimate_strehl(self):
        """
        Estimates strehl, using Markos van Dam's accurate Strehlometer
        """
        inputdata = self.current_image
        if self.current_image is None:
            inputdata = self.previous_image
        return vDs.strehl(np.array(inputdata.shaped),
                          np.array(self.ref_psf.shaped))
