import hcipy
import numpy as np
from common import support_functions as sf
from common import classes as ff_c
import ipdb
comment = """
[SIMULATION]
    run_sim         = boolean(default=False)
    flux            = float(min=0)
    exptime         = float(min=0)
    rms_wfe         = float(min=0)
    seed            = integer
"""
def center_image(small_image, large_size, center_position):
    """Centers a smaller image at a specific position in a larger empty image

    Parameters
    ----------
    small_image : ndarray
        The smaller image to be centered
    large_size : tuple
        Size of the larger image (height, width)
    center_position : tuple
        Position to center the small image (y, x)

    Returns
    -------
    ndarray
        The larger image with the small image centered at the specified position
    """
    # Create empty large image
    large_image = np.zeros(large_size, dtype=small_image.dtype)

    # Calculate corners for placement
    x_start = center_position[0] - small_image.shape[0]//2
    x_end = x_start + small_image.shape[0]
    y_start = center_position[1] - small_image.shape[1]//2
    y_end = y_start + small_image.shape[1]

    # Place small image in large image
    large_image[y_start:y_end, x_start:x_end] = small_image

    return large_image


class FakeCoronagraphOpticalSystem:
    """A helper class to build a coronagraph from a configuration file"""
    def __new__(self, **optical_params):
        
        self.Npix_pup = optical_params['N pix pupil']
        self.Npix_foc = optical_params['N pix focal']
        self.pixscale = optical_params['pixel scale (mas/pix)']
        self.wavelength = optical_params['wavelength (m)']
        #rots and flips applied last
        self.flipx = optical_params['flip_x']
        self.flipy = optical_params['flip_y']
        
        self.rotation_angle_deg = optical_params['rotation angle im (deg)']
        self.aperturename = optical_params['APERTURE']['aperture']
        self.rotation_angle_aperture = optical_params['APERTURE']['rotation angle aperture (deg)']
        
        self.coronagraph_IWA_mas = optical_params['CORONAGRAPH_MASK']['IWA_mas']
        
        self.lyotstopmask = optical_params['LYOT_STOP']['lyot stop']
        self.rotation_angle_lyot = optical_params['LYOT_STOP']['rotation angle lyot (deg)']

        self.TelescopeAperture = ff_c.Aperture(Npix_pup=self.Npix_pup,
                                               aperturename=self.aperturename,
                                               rotation_angle_aperture=self.rotation_angle_aperture)
        self.Lyotcoronagraph   = ff_c.LyotCoronagraph(Npix_foc=self.Npix_foc, 
                                                      IWA_mas=self.coronagraph_IWA_mas, 
                                                      mas_pix=self.pixscale, 
                                                      pupil_grid=self.TelescopeAperture.pupil_grid)
        self.LyotStop          = ff_c.Aperture(Npix_pup=self.Npix_pup,
                                               aperturename=self.lyotstopmask,
                                              rotation_angle_aperture = self.rotation_angle_lyot)
        
        self.CSM = ff_c.CoronagraphSystemModel(telescopeaperture=self.TelescopeAperture,
                           coronagraph=self.Lyotcoronagraph,
                           lyotaperture=self.LyotStop,
                           Npix_foc=self.Npix_foc,
                           mas_pix=self.pixscale,
                           wavelength=self.wavelength, 
                           flipx=self.flipx,
                           flipy=self.flipy,
                           rotation_angle_deg=self.rotation_angle_deg)
        return self.CSM
    def __init__(self):
        pass

class FakeDetector:
    """A class to simulate a fake camera.  Accepts and optical system
    and pulls the latest electric field when you run take_image()
    Parameters
    ---------
    input_grid - hcipy Grid
        the grid onto which to project the image
    read_noise - float
        e-/pix/read
    dark_current_rate - float
        e-/pix/s
    flat_field - float or array
        Not sure how this works
    include_photon_noise - Bool
        whether it will add poisson noise to the flux
    exptime - float
        exposure time in seconds
    optical system - classes.py SystemModel
    """

    def __init__(self,
                 flux = None,
                 read_noise=0,
                 dark_current_rate=0,
                 flat_field=0,
                 bias_offset=0,
                 include_photon_noise=True,
                 exptime=None,
                 xsize = None,
                 ysize = None,
                 field_center_x = None,
                 field_center_y = None,
                 flip_x = None,
                 flip_y = None,
                 rotation_angle_deg = None, #not yet implemented
                 opticalsystem=None):
        self.flux = flux
        self.input_grid = opticalsystem.focal_grid
        self.read_noise = read_noise
        self.dark_current_rate = dark_current_rate
        self.include_photon_noise = include_photon_noise
        self.flat_field = flat_field
        self.bias_offset = bias_offset
        self.xsize = xsize
        self.ysize = ysize
        self.field_center_x = field_center_x
        self.field_center_y = field_center_y
        self.flip_x = flip_x #THESE SHOULD BE REMOVED--NOT IMPLEMENTED.  SHOULD BE IN OPTICAL SYSTEM.
        self.flip_y = flip_y
        self.rotation_angle_deg = rotation_angle_deg

        # passed by reference...so the latest efields will update
        self.opticalsystem = opticalsystem
        self.exptime = exptime

        self.detector = hcipy.optics.NoisyDetector(
                              self.input_grid,
                              dark_current_rate=self.dark_current_rate,
                              read_noise=self.read_noise,
                              flat_field=0,
                              include_photon_noise=self.include_photon_noise)

        return

    def take_image(self, focal_wf=None, t=None):
        """Returns an image.
        focal_wf - hcipy wavefront
            the focal plane wavefront.  If None, will pull
            the opticalsystem.focal_efield
        Parameters
        ----------
        focal_wf - hcipy Wavefront
            focal plane wavefront, used to calculate the power
            and hence energy in time t
        t - float
            integration time in seconds

        Returns
        -------
        """
        if focal_wf is None:
            this_focal_wf = self.opticalsystem.focal_efield.copy()
        else:
            this_focal_wf = focal_wf.copy()
        if t is None:
            t = self.exptime
        this_focal_wf.total_power = self.flux
        self.detector.integrate(this_focal_wf, t)
        # convert to numpy array like the camera would deliver
        hcipy_image = np.array(self.detector.read_out().shaped)
        output_image = center_image(hcipy_image, (self.xsize, self.ysize),
                                    (self.field_center_x, self.field_center_y))
        output_image += np.random.poisson(self.dark_current_rate*self.exptime, size=output_image.shape)
        output_image += np.random.normal(0, self.read_noise, size=output_image.shape)
        output_image += self.bias_offset
        return output_image

class FakeAOSystem:
    """A class that has the same API as the normal AO system class.
       Accepts an optical system and modifies the pupil efield by reference"""
    def __init__(self, OpticalModel,
                       modebasis=None,
                       initial_rms_wfe=0,
                       rotation_angle_dm = 0,
                       seed=None):
        if seed is not None:
            np.random.seed(seed)
        print("Warning: rotation angle dm is not implemented yet in the simulator")
        self.OpticalModel = OpticalModel
        self.initial_phase_error = sf.generate_random_phase(rms_wfe=initial_rms_wfe,
                                                            mode_basis=modebasis,
                                                            pupil_grid=self.OpticalModel.Pupil.pupil_grid,
                                                            aperture=self.OpticalModel.Pupil.aperture)
        self.OpticalModel.update_pupil_wavefront(self.initial_phase_error)

    def set_dm_data(self, dm_microns):
        #in the sim, just undoes the microns command from subaru
        phase_DM= dm_microns / self.OpticalModel.wavelength * (2 * np.pi) / 1e6
        self.OpticalModel.update_pupil_wavefront(self.initial_phase_error-phase_DM)
        self.OpticalModel.generate_psf_efield()
        return

    def make_dm_command(self, microns):
        return microns

    def close_dm_stream(self):
        return


class FakeAODMSystem:
    """A class that has the same API as the normal AO system class.
    Accepts an optical system and modifies the pupil efield by reference

    Really just `FakeAOSystem,` but with an hcipy deformable mirror model
    """
    def __init__(self, OpticalModel,
                       modebasis=None,
                       initial_rms_wfe=0,
                       rotation_angle_dm = 0,
                       num_actuators_across=22,
                       actuator_spacing=None,
                       seed=None):

        if seed is not None:
            np.random.seed(seed)
        
        print("Warning: rotation angle dm is not implemented yet in the simulator")
        self.num_actuators_across = num_actuators_across
        self.OpticalModel = OpticalModel
        self.initial_phase_error = sf.generate_random_phase(rms_wfe=initial_rms_wfe,
                                                            mode_basis=modebasis,
                                                            pupil_grid=self.OpticalModel.Pupil.pupil_grid,
                                                            aperture=self.OpticalModel.Pupil.aperture)
        self.OpticalModel.update_pupil_wavefront(self.initial_phase_error)
        self.modebasis = modebasis
        self.rotation_angle_dm = rotation_angle_dm

        # Build deformable mirror
        if actuator_spacing is None:

            # compute based on num actuators across and pupil diameter
            actuator_spacing = self.OpticalModel.Pupil.diameter / num_actuators_across
        
        self.influence_functions = hcipy.make_gaussian_influence_functions(self.OpticalModel.Pupil.pupil_grid,
                                                                           num_actuators_across,
                                                                           actuator_spacing)
        self.deformable_mirror = hcipy.DeformableMirror(self.influence_functions)

    def set_dm_data(self, dm_commands, modify_existing=False):
        """
        NOTE: Not actually sure that this is the right shape
        Parameters
        ----------
        dm_commands : ndarray
            modification to DM actuator heights in an array of shape
            Nactuators x Nactuators units of meters.
        """
        assert dm_commands.shape[0] == self.num_actuators_across
        assert dm_commands.shape[1] == self.num_actuators_across

        #in the sim, just undoes the microns command from subaru
        phase_DM_acts = dm_commands / self.OpticalModel.wavelength * (2 * np.pi) 
        print(f"Wavelength = {self.OpticalModel.wavelength}")
        # Modify existing DM surface
        if modify_existing:
            self.deformable_mirror.actuators += phase_DM_acts.ravel()
        else:
            self.deformable_mirror.actuators = phase_DM_acts.ravel()

        phase_DM = self.deformable_mirror.opd
        self.phase_DM = phase_DM
        self.OpticalModel.update_pupil_wavefront(self.initial_phase_error-phase_DM)
        self.OpticalModel.generate_psf_efield()

        return

    def make_dm_command(self, microns):
        return microns

    def close_dm_stream(self):
        return
