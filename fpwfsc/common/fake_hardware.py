import hcipy
import numpy as np
import sys
import os
import time
import astropy.io.fits as pf
from pathlib import Path

from . import support_functions as sf
from . import classes as ff_c
from . import dm
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

def create_bad_pixel_mask(height, width, bad_pixel_fraction, outputfile=None):
    """Creates a random bad pixel mask.

    Parameters
    ----------
    height : int
        Height of the detector in pixels
    width : int
        Width of the detector in pixels
    bad_pixel_fraction : float
        Fraction of pixels that should be marked as bad (between 0.0 and 1.0)

    Returns
    -------
    ndarray
        Boolean mask where True indicates a bad pixel
    """
    if bad_pixel_fraction <= 0:
        return None

    bad_pixel_mask = np.zeros((height, width), dtype=bool)
    num_bad_pixels = int(bad_pixel_fraction * height * width)

    # Randomly select pixels to mark as bad
    bad_indices = np.random.choice(height * width,
                                  size=num_bad_pixels,
                                  replace=False)
    bad_y, bad_x = np.unravel_index(bad_indices, (height, width))
    bad_pixel_mask[bad_y, bad_x] = True

    if outputfile is not None:
        pf.writeto(outputfile, 1*bad_pixel_mask, overwrite=True)

    return bad_pixel_mask

class HitchhikerTimeoutError(Exception):
    """Raised when Hitchhiker times out waiting for a new image."""
    pass


class Hitchhiker:
    """
    Watches a directory for new FITS files and loads them once they appear fully written.
    Assumes all files will match the size of the first complete file.

    Args:
        imagedir (str or Path): Directory to monitor for incoming FITS files.
        poll_interval (float): Time (in seconds) between file system checks.
        timeout (float): Maximum time (in seconds) to wait for a new file.
    """

    def __init__(self, imagedir=None, poll_interval=0.5, timeout=20):
        self.watch_dir = Path(imagedir)
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.suffixes = ('.fits', '.fit')
        self.seen_files = set()
        self.reference_size = None

        if not self.watch_dir.exists():
            raise FileNotFoundError(f"Directory {self.watch_dir} does not exist.")

    def wait_for_next_image(self, timeout=None):
        """
        Waits for the next complete FITS file to appear in the watch directory.

        Args:
            timeout (float or None): Override the default timeout set at initialization.

        Returns:
            numpy.ndarray: Image data from the FITS file.

        Raises:
            HitchhikerTimeoutError: If no file is found within the timeout period.
        """
        timeout = timeout or self.timeout
        start_time = time.time()

        while True:
            new_files = sorted([
                f for f in self.watch_dir.iterdir()
                if f.suffix.lower() in self.suffixes and f not in self.seen_files
            ], key=os.path.getmtime)

            for f in new_files:
                size = f.stat().st_size
                if size == 0:
                    continue

                if self.reference_size is None:
                    if self._is_fully_written(f):
                        self.reference_size = f.stat().st_size
                        self.seen_files.add(f)
                        return self._read_fits(f)
                    else:
                        continue

                if size == self.reference_size:
                    self.seen_files.add(f)
                    return self._read_fits(f)
                else:
                    print(f"⚠️  File {f.name} has size {size}, expected {self.reference_size}. Trying to open anyway...")
                    try:
                        self.seen_files.add(f)
                        return self._read_fits(f)
                    except Exception as e:
                        print(f"Could not read {f.name}: {e}")
                        continue

            if timeout and (time.time() - start_time) > timeout:
                raise HitchhikerTimeoutError("No valid FITS file found within timeout.")

            time.sleep(self.poll_interval)

    def _is_fully_written(self, filepath):
        """Check if a file has stopped growing, to confirm it's finished writing."""
        try:
            size1 = filepath.stat().st_size
            time.sleep(0.2)
            size2 = filepath.stat().st_size
            return size1 == size2 and size1 > 0
        except FileNotFoundError:
            return False

    def _read_fits(self, filepath):
        """Reads and returns the primary HDU data from a FITS file."""
        with pf.open(filepath, ignore_missing_end=True, do_not_scale_image_data=True) as hdul:
            return hdul[0].data


class FakeCoronagraphOpticalSystem:
    """A helper class to build a coronagraph from a configuration file"""
    def __new__(self, include_fpm=True, **optical_params):

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

        self.include_fpm = include_fpm
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
                           rotation_angle_deg=self.rotation_angle_deg,
                           include_fpm=self.include_fpm)
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
             bad_pixel_mask=None,
             bias_offset=0,
             include_photon_noise=True,
             exptime=None,
             xsize = None,
             ysize = None,
             field_center_x = None,
             field_center_y = None,
             rotation_angle_deg = None, #not yet implemented
             opticalsystem=None,
             output_directory=None):  
        self.flux = flux
        self.input_grid = opticalsystem.focal_grid
        self.read_noise = read_noise
        self.dark_current_rate = dark_current_rate
        self.include_photon_noise = include_photon_noise
        self.flat_field = flat_field
        self.bad_pixel_mask = bad_pixel_mask
        self.bias_offset = bias_offset
        self.xsize = xsize
        self.ysize = ysize
        self.field_center_x = field_center_x
        self.field_center_y = field_center_y
        self.rotation_angle_deg = rotation_angle_deg
        self.output_directory = output_directory  # Store the output directory
    
        # passed by reference...so the latest efields will update
        self.opticalsystem = opticalsystem
        self.exptime = exptime
        if self.bad_pixel_mask is not None:
            self.badpixelmask = np.array(pf.open(self.bad_pixel_mask)[0].data, dtype=bool)
            self.nbadpix = np.sum(self.badpixelmask)

        self.detector = hcipy.optics.NoisyDetector(
                              self.input_grid,
                              dark_current_rate=self.dark_current_rate,
                              read_noise=0,
                              #this is a hack to deal with not double counting read noise when making the image larger
                              flat_field=0,
                              include_photon_noise=self.include_photon_noise)

        return

    def take_image(self, focal_wf=None, t=None, output_directory=None):
        """Returns an image and optionally saves it to a directory.
    
        Parameters
        ----------
        focal_wf : hcipy Wavefront, optional
            focal plane wavefront, used to calculate the power
            and hence energy in time t
        t : float, optional
            integration time in seconds
        output_directory : str, optional
            directory path where the image should be saved as a FITS file.
            If None, uses the output_directory set during initialization.

        Returns
        -------
        ndarray
            The generated image
        """
        # Use provided output_directory or fall back to self.output_directory
        output_directory = output_directory or self.output_directory
    
        # Generate the image
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
        if self.bad_pixel_mask is not None:
            output_image[self.badpixelmask] = np.random.uniform(0.9, 1.1, size=self.nbadpix)*100*np.std(output_image)

        # Save image to disk if a directory is specified
    
        if output_directory is None:
            return output_image

        try:
            os.makedirs(output_directory, exist_ok=True)

            # Generate a unique filename (appending a counter if needed)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            base_filename = f"image_{timestamp}"
            filename = f"{base_filename}.fits"
            filepath = os.path.join(output_directory, filename)
            counter = 1
            while os.path.exists(filepath):
                filename = f"{base_filename}_{counter}.fits"
                filepath = os.path.join(output_directory, filename)
                counter += 1

            print(f"Saving image to {filepath}...")
            with open(filepath, 'wb') as f:
                pf.writeto(f, output_image, overwrite=True)
                f.flush()
                os.fsync(f.fileno())  # Ensures it's really written to disk

            print(f"Image saved to {filepath}")
            return output_image

        except Exception as e:
            print(f"Error saving image to disk: {e}")
            return None

    
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
                       num_actuators_across=21,
                       actuator_spacing=None,
                       seed=None,
                       flip_x_dm=None,
                       flip_y_dm=None):

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

        # This is the 'Clocking' of the DM with respect to the image
        self.rotation_angle_dm = rotation_angle_dm
        self.flip_x_dm = flip_x_dm
        self.flip_y_dm = flip_y_dm

        # Build deformable mirror
        if actuator_spacing is None:

            # compute based on num actuators across and pupil diameter
            actuator_spacing = self.OpticalModel.Pupil.diameter / num_actuators_across

        self.influence_functions = hcipy.make_gaussian_influence_functions(self.OpticalModel.Pupil.pupil_grid,
                                                                           num_actuators_across,
                                                                           actuator_spacing)
        self.deformable_mirror = hcipy.DeformableMirror(self.influence_functions)

        self.current_dm_shape = np.zeros([self.num_actuators_across,
                                          self.num_actuators_across])

    def get_dm_data(self):
        return self.current_dm_shape


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
        #print(f"Wavelength = {self.OpticalModel.wavelength}")
        # Modify existing DM surface
        if modify_existing:
            self.current_dm_shape += dm_commands
            self.deformable_mirror.actuators += phase_DM_acts.ravel()
        else:
            self.current_dm_shape = dm_commands
            self.deformable_mirror.actuators = phase_DM_acts.ravel()


        # NOTE: Deformable_mirror.opd doesn't add the negative OPD you get from hitting a mirror, so the minus
        # passed to phase_DM in update_pupil_wavefront takes care of it
        phase_DM = self.deformable_mirror.opd
        self.phase_DM = phase_DM
        phase_DM = sf.rotate_and_flip_field(phase_DM,
                                                angle=self.rotation_angle_dm,
                                                flipx=self.flip_x_dm,
                                                flipy=self.flip_y_dm)
        self.OpticalModel.update_pupil_wavefront(self.initial_phase_error - phase_DM)
        self.OpticalModel.generate_psf_efield()

        return

    def offset_tiptilt(self, metersx, metersy,
                             angle=None,
                             flipx=None,
                             flipy=None):
        pass

    def make_dm_command(self, microns):
        return microns

    def close_dm_stream(self):
        return


if __name__ == "__main__":
    pass
