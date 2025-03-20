import hcipy
import numpy as np
from common import support_functions as sf

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
        self.flip_x = flip_x
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
        self.OpticalModel = OpticalModel
        self.initial_phase_error = sf.generate_random_phase(rms_wfe=initial_rms_wfe,
                                                            mode_basis=modebasis,
                                                            pupil_grid=self.OpticalModel.Pupil.pupil_grid,
                                                            aperture=self.OpticalModel.Pupil.aperture)
        self.OpticalModel.update_pupil_wavefront(self.initial_phase_error)

        # Build deformable mirror
        if actuator_spacing is None:

            # compute based on num actuators across and pupil diameter
            actuator_spacing = self.OpticalModel.Pupil.diameter / num_actuators_across
        
        self.influence_functions = hcipy.make_gaussian_influence_functions(self.OpticalModel.Pupil.pupil_grid,
                                                                           num_actuators_across,
                                                                           actuator_spacing)
        self.deformable_mirror = hcipy.DeformableMirror(self.influence_functions)

    def set_dm_data(self, dm_microns, modify_existing=False):
        """
        NOTE: Not actually sure that this is the right shape
        Parameters
        ----------
        dm_microns : ndarray
            modification to DM actuator heights in an array of shape
            Nactuators x Nactuators units of microns.
        """

        #in the sim, just undoes the microns command from subaru
        phase_DM_acts = dm_microns / self.OpticalModel.wavelength * (2 * np.pi) / 1e6

        # Modify existing DM surface
        if modify_existing:
            self.deformable_mirror.actuators += phase_DM_acts
        else:
            self.deformable_mirror.actuators = phase_DM_acts

        phase_DM = self.deformable_mirror.opd


        self.OpticalModel.update_pupil_wavefront(self.initial_phase_error-phase_DM)
        self.OpticalModel.generate_psf_efield()

        return

    def make_dm_command(self, microns):
        return microns

    def close_dm_stream(self):
        return
