import os
import sys
import numpy as np
from scipy.ndimage import affine_transform, median_filter
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import json
import io
from configobj import flatten_errors, ConfigObj
from validate import Validator, ValidateError
import warnings
import hcipy
import ipdb


# NOTE: These flips are used for the simulation
#def reduce_images(data, npix=None, refpsf=None, xcen=None, ycen=None, bgds=None, flipx=False, flipy=False):
# NOTE: These flips are used when running with NIRC2
def reduce_images(data, npix=None, refpsf=None, xcen=None, ycen=None, bgds=None, flipx=False, flipy=False, rotation_angle_deg=0):
    """This function is modified from take_images() to only reduce
    a given output with the following steps:

    1) Crops the array to a square of npix x npix
    2) Preprocesses the array to bgd sub, remove bad pix, flatfield
    3) Aligns it with a reference PSF.

    Inputs
    --------
    data: object
        image from a science detector
    npix: integer
        size to crop image to, that is, (npix x npix)
    refpsf: numpy array
        reference psf in units of intensity (not efield)
        must be same size as cropped size (npix, npix)
    xcen: integer
        x center of the pre-cropped image
    ycen: integer
        y center of the pre-cropped image
    bgds: dictionary
        a dictionary containing ndarrays the same size as the raw
        image of the following:
        'bkgd' = the background,
        'masterflat'= the flat field (values around 1)
        'badpix'=the bad pixel map (bad pixels = 1, other=0)
    """
    print('\nTaking data')
    if npix is None:
        npix = refpsf.shape[0]
    if xcen is None:
       xcen = npix//2
    if ycen is None:
       ycen = npix//2

    # data cube that will contain all the images that will be averaged
    data_cube = np.zeros(( npix, npix))
    #0 crop backgrounds to required size
    if bgds is None:
        bgds={'bkgd':None, 'badpix':None, 'masterflat':None}

    if bgds['bkgd'] is not None:
        bkgd_crop    = square_crop(bgds['bkgd'], npix, xcen, ycen)
    else: bkgd_crop = None
    if bgds['badpix'] is not None:
        badpix_crop = square_crop(bgds['badpix'], npix, xcen, ycen)
    else: badpix_crop = None
    if bgds['masterflat'] is not None:
        flat_crop   = square_crop(bgds['masterflat'], npix, xcen, ycen)
    else: flat_crop = None

    raw_image = data
    #1 crop step

    cropped_image = square_crop(raw_image, npix, xcen, ycen)
    #2 basic preprocessing step
    eq_image = equalize_image(cropped_image,
                              bkgd=bkgd_crop,
                              masterflat=flat_crop,
                              badpix=badpix_crop)
    #align to ref psf
    if refpsf is not None:
        offset = phase_cross_correlation(refpsf, eq_image,
                                         upsample_factor=10, normalization=None)[0]
        shifted = shift(eq_image, offset, order=3, mode='wrap')
    else:
        shifted = eq_image
    #coadd
    final_image = shifted

    if abs(rotation_angle_deg) > 1e-6:
        final_image = cen_rot(final_image, rotation_angle_deg, np.array([final_image.shape[0]/2,final_image.shape[1]/2]))

    if flipx:
        final_image = np.flip(final_image, axis=0)

    if flipy:
        final_image = np.flip(final_image, axis=1)

    return final_image
'''
def take_images(detector=None, n_images=1, npix=None,
                refpsf=None, xcen=None, ycen=None, bgds=None):
    """This is a `super` function that induces the detector class
    to take one or more images, then performs the following steps:

    1) Crops the array to a square of npix x npix
    2) Preprocesses the array to bgd sub, remove bad pix, flatfield
    3) Aligns it with a reference PSF.
    4) Repeats this N times and stacks the result

    Inputs
    --------
    detector: object
        class that controls the camera.  needs to have a method called
        take_image() that returns a single image
    n_images: integer
        number of images to take, defaults to 1
    npix: integer
        size to crop image to, that is, (npix x npix)
    refpsf: numpy array
        reference psf in units of intensity (not efield)
        must be same size as cropped size (npix, npix)
    xcen: integer
        x center of the pre-cropped image
    ycen: integer
        y center of the pre-cropped image
    bgds: dictionary
        a dictionary containing ndarrays the same size as the raw
        image of the following:
        'bkgd' = the background,
        'masterflat'= the flat field (values around 1)
        'badpix'=the bad pixel map (bad pixels = 1, other=0)
    """
    print('\nTaking data')
    if npix is None:
        npix = refpsf.shape[0]
    if xcen is None:
       xcen = npix//2
    if ycen is None:
       ycen = npix//2

    # data cube that will contain all the images that will be averaged
    data_cube = np.zeros((n_images, npix, npix))
    #0 crop backgrounds to required size
    if bgds['bkgd'] is not None:
        bkgd_crop    = square_crop(bgds['bkgd'], npix, xcen, ycen)
    else: bkgd_crop = None
    if bgds['badpix'] is not None:
        badpix_crop = square_crop(bgds['badpix'], npix, xcen, ycen)
    else: badpix_crop = None
    if bgds['masterflat'] is not None:
        flat_crop   = square_crop(bgds['masterflat'], npix, xcen, ycen)
    else: flat_crop = None
    for i in np.arange(n_images):
        raw_image = detector.take_image()
        #1 crop step
        cropped_image = square_crop(raw_image, npix, xcen, ycen)
        #2 basic preprocessing step
        eq_image = equalize_image(cropped_image,
                                  bkgd=bkgd_crop,
                                  masterflat=flat_crop,
                                  badpix=badpix_crop)
        #align to ref psf
        offset = phase_cross_correlation(refpsf, eq_image,
                                         upsample_factor=10)[0]
        shifted = shift(eq_image, offset, order=3, mode='wrap')

        data_cube[i,:,:] = shifted
    #coadd
    final_image = np.sum(data_cube, axis=0)
    return final_image
'''

def removebadpix(data, mask, kern = 5):
    """Removes bad pixels by replacing them with a median-filtered
    version of the image.  The slow part here is computing the median,
    this can be sped up by iterating over pixels in my opinion.
    Inputs
    -------
        data--a 2d numpy array
        mask--a 2d numpy binary mask indicating bad pixels
              (ones are bad)
        kern--the kernel to compute the median filter
    Outputs
    -------
        a 2d numpy array with the bad pixels replaced by the median
    """

    # Create a copy of the provided data
    tmp = data.copy()
    # Compute the medfilt image associated to the provided data
    medianed_image = median_filter(tmp, size=(kern, kern), mode='wrap')
    # Replaces the bad pixel by the computed value
    tmp[np.where(mask>0)] = medianed_image[np.where(mask>0)]
    # Return the cleaned data
    return tmp

def equalize_image(data, bkgd=None, masterflat=None, badpix=None):
    """removes bad pixels from bgd-subtracted data,
    and divides by the master flat field
    Inputs
    -------
    data - 2d np array
        your data
    bgd - 2d np array
        backgrounds, defaults to median of data if None
    masterflat - 2d np array
        the flatfield, defaults to 1 if None
    badpix - 2d binary np array
        the bad pixel map, 1 where bad pixels exist
        if None, ignored

    Returns - 2d np array
        the cleaned image
    """
    if bkgd is None:
        bkgd = np.median(data)
    if masterflat is None:
        masterflat = 1
    if badpix is None:
        return (data-bkgd)/masterflat
    else:
        return removebadpix(data-bkgd, badpix)/masterflat

def square_crop(image, npix, xcen, ycen):
    """Crop an image to npix x npix, about the point xcen, ycen."""
    #this makes sure the image size is correct for odd sizes
    if npix % 2 == 1:
        bonus = 1
    else:
        bonus = 0
    hw     = npix//2
    startx = xcen - hw
    endx   = xcen + hw
    starty = ycen - hw
    endy   = ycen + hw
    return image[(starty-bonus):endy, (startx-bonus):endx]

class MyValidator(Validator):
    def __init__(self):
        super().__init__()
        self.functions['float_or_none'] = self._float_or_none
        self.functions['integer_or_none'] = self._integer_or_none
        self.functions['option_or_none'] = self._option_or_none


    def _float_or_none(self, value, *args):
    # Remove the debugging breakpoint
    # ipdb.set_trace()
    
        # Handle the case when value is a list (which appears to be happening)
        if isinstance(value, list):
            # If it's a list containing option specifications, return None
            # This is likely a parsing issue in the config system
            if any('None' in str(item) for item in value):
                return None
            # Try to convert the first element if it's a simple list
            try:
                return float(value[0])
            except (ValueError, IndexError):
                raise ValidateError("Expected float or 'None', got list: {}".format(value))
    
        # Original logic for string values
        if value in ('None', ''):
            return None
        try:
            return float(value)
        except ValueError:
            raise ValidateError("Expected float or 'None'")

    
    def _integer_or_none(self, value, *args):
        # Handle the case when value is a list
        if isinstance(value, list):
            # If it's a list containing option specifications, return None
            if any('None' in str(item) for item in value):
                return None
            # Try to convert the first element if it's a simple list
            try:
                return int(value[0])
            except (ValueError, IndexError):
                raise ValidateError("Expected int or 'None', got list: {}".format(value))

        # Original logic for string values
        if value in ('None', ''):
            return None
        try:
            return int(value)
        except ValueError:
            raise ValidateError("Expected int or 'None'")
    
    def _option_or_none(self, value, *args):
        """Validates that a value is either None or one of the specified options"""
        if value in ('None', '', None):
            return None
            
        # Convert the args to strings for comparison
        args = [str(arg) for arg in args]
        
        if str(value) in args:
            return value
        else:
            raise ValidateError("Value must be one of: None, {0}".format(', '.join(args)))


def validate_config(config_input, configspec=None):
    """A helper function that checks a config input for errors against
    the configspec file. Accepts both filenames and dictionaries.
    
    Parameters
    ---------
    config_input - string or dict
        the configuration to check (filename or dictionary)
    configspec - string
        the configspec file (eg, config.spec)

    Returns
        If passed, the config object
        If failed, prints the errors
    """
    config = ConfigObj(config_input, configspec=configspec)
    val = MyValidator()
    res = config.validate(val, preserve_errors=True)
    
    input_type = "Config file" if isinstance(config_input, str) else "Config dictionary"
    
    if res is True:
        print(f"{input_type} PASSED VALIDATION CHECK")
        return config
    else:
        print(f"{input_type} FAILED VALIDATION CHECK")
        print(flatten_errors(config, res))
        sys.exit(0)



def orthonormalize_mode_basis(mode_basis, aperture, epsilon=1E-2):
    """Orthonormalize a mode basis created by hcipy over an aperture

    Parmeters
    --------
    mode_basis - hcipy mode_basis
        the input mode basis
    aperture - hcipy  Field ?
        The aperture over which to normalize
    epsilon - a parameter to clip the aperture below, to 0

    Returns
    -------
    mode_basis_orthogonal - hcipy mode_basis
        The orthogonalized mode basis
    """
    # making sure that we have othogonal modes
    mode_basis_temp = hcipy.ModeBasis(
                        transformation_matrix=mode_basis.transformation_matrix *
                        (aperture > epsilon)[:,np.newaxis], grid=mode_basis.grid)
    mode_basis_orthogonal = mode_basis_temp.orthogonalized

    for mode in mode_basis_orthogonal:
        rms_mode = rms(mode, aperture)

        if rms_mode < 1E-8:
            mode /= np.mean(mode[aperture>epsilon])

        else:
            mode /= rms_mode
    return mode_basis_orthogonal

def fouriersplit(p, fourier_transform):
    ''' Decomposes a given (complex) array p into its odd and even constituents using FT symmetry

    NB - Can be extended to arbitrary complex array

    Based on the code of M. Wilby.

    Parameters
    ----------
    p : hcipy.Field
        The field that will be decomposed in its odd and even constituents.
    fourier_transform : hcipy FourierTransform object
        The Fourier transform.

    Returns
    -------
    p_e : hcipy.Field
        The even part of p.
    p_o : hcipy.Field
        The odd part of p.
    '''
    # taking the Fourier transform of the real and imaginary parts of p.
    # Split by real/imaginary part of input
    P_r = fourier_transform.forward((p.real).astype(np.complex128))
    P_i = fourier_transform.forward((1j * p.imag).astype(np.complex128))

    # inverse Fourier transform to get back the even and odd parts of p.
    p_e = fourier_transform.backward(P_r.real + 1j * P_i.imag) # Even complex array
    p_o = fourier_transform.backward(1J * P_r.imag + P_i.real) # Odd complex array

    return p_e, p_o

def rms(phase, aperture):
    #Calculates the RMS of the phase.
    temp_phase = np.array(phase)
    RMS = np.sqrt(np.mean((temp_phase[aperture > 0] - np.mean(temp_phase[aperture > 0])) ** 2))
    return RMS

def check_directory(path):
    '''
    This function checks if the directory is already there, if not it will create a
    new directory.
    '''

    if not os.path.exists(path):
        os.makedirs(path)
    return

def modal_decomposition(phase, basis):
    """Decomposes a wavefront on a certain basis."""
    coeffs = np.dot(hcipy.inverse_truncated(basis.transformation_matrix), phase)
    #coeffs = np.dot(basis.transformation_matrix, phase)
    return coeffs

def modal_recomposition(modal_coeffs, mode_basis, aperture):
    """Dots a wavefront into a particular basis set"""
    phase = hcipy.Field(np.sum(modal_coeffs[:, np.newaxis] * \
                        np.array(mode_basis), axis=0),
                        aperture.grid)
    return phase

def remove_piston(phase, aperture):
    """Remove piston from a wavefront over an aperture"""
    retphase = phase.copy()
    retphase[aperture] = retphase[aperture] - np.mean(retphase[aperture])
    return retphase

def fourier_resample_v2(input_data, new_shape, output_diam = None):
    rawdataframe = input_data.copy().shaped.astype(np.complex128)
    # ()
    input_diam = (input_data.grid.x[0]-(input_data.grid.x[2]-input_data.grid.x[1])/2)*(-2)

    # Scales the frame with scalefactor
    Lenx, Leny = rawdataframe.shape
    grid_from  = hcipy.make_pupil_grid((Lenx, Leny), [input_diam, input_diam])
    grid_to    = hcipy.make_pupil_grid((new_shape[0], new_shape[1]), [output_diam, output_diam])

    fft = hcipy.FastFourierTransform(grid_from, q=1, fov=1)
    mft = hcipy.MatrixFourierTransform(grid_to, fft.output_grid)
    reshaped = mft.backward(fft.forward(hcipy.Field(rawdataframe.ravel(), grid_from)))

    return  reshaped.real

def fourier_resample(input_data, new_shape):
    rawdataframe = input_data.copy().shaped.astype(np.complex128)

    # Scales the frame with scalefactor
    Lenx, Leny = rawdataframe.shape
    grid_from  = hcipy.make_pupil_grid((Lenx, Leny))
    grid_to    = hcipy.make_pupil_grid((new_shape[0], new_shape[1]))

    fft = hcipy.FastFourierTransform(grid_from, q=1, fov=1)
    mft = hcipy.MatrixFourierTransform(grid_to, fft.output_grid)
    reshaped = mft.backward(fft.forward(hcipy.Field(rawdataframe.ravel(), grid_from)))

    return  reshaped.real

def rotate_and_flip_wavefront(hcipy_wavefront, angle=None, flipx=False, flipy=False):
    """Rotate and flip a hcipy field
    angle in degrees
    flip_x and flip_y are booleans
    """
    wf = hcipy_wavefront.copy()
    wf_efield = wf.electric_field
    #First rotate about center
    if np.abs(angle)>1e-6:
        center = np.array(wf_efield.shaped.shape)/2
        rotated_field = cen_rot(wf_efield.shaped, angle, center)
        final_field = rotated_field
    else:
        final_field = wf_efield.shaped
    if flipx:
        final_field = np.flip(final_field, axis=0)

    if flipy:
        final_field = np.flip(final_field, axis=1)

    wf.electric_field = hcipy.Field(final_field.ravel(), grid=wf.grid)
    return wf

def cen_rot(im, rot, rotation_center):
    '''
    cen_rot - takes a cube of images im, and a set of rotation angles in rot,
    and translates the middle of the frame with a size dim_out to the middle of
    a new output frame with an additional rotation of rot.
    '''
    # converting rotation to radians
    a = np.radians(rot)

    # make a rotation matrix
    transform = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])[:,:]
    # calculate total offset for image output

    c_in = rotation_center#center of rotation
    # c_out has to be pre-rotated to make offset correct

    offset = np.dot(transform, -c_in) + c_in
    offset = (offset[0], offset[1],)
    # perform the transformation
    dst = affine_transform(im, transform, offset=offset)

    return dst

def save_dict(dicty, save_name):
    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    # Write JSON file
    with io.open(save_name, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(dicty,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))

def load_dict(load_name):
     # Read JSON file

    with open(load_name) as data_file:
        dict_loaded = json.load(data_file)

    return dict_loaded

def calculate_VAR(image, reference_PSF, mas_pix, wavelength, diameter):
    ''' Calculates the Variance of the normalized first Airy Ring (VAR) of the image.

    For the exact definition, see equation 13 in Bos et al. (2020).

    Parameters
    ----------
    image : HCIPy Field object
        The image containing the PSF of the system, centered to the reference PSF.
    reference_PSF : HCIPy hcipy.Field object
        The numerically calculated reference PSF.
    mas_pix : float
        The pixel scale of the detector in milliarcsec per pixel.
    wavelength : float
        The central wavelength of the filter operated in, provided in meter.
    diameter : float
        The projected diameter of the exit pupil in meter.
    '''
    # Calculating what lambda/D is in milliarcsec.
    lambda_D_rad = wavelength / diameter
    lambda_D_mas = np.degrees(lambda_D_rad) * 3600 * 1000

    # number of pixels along one axis in the focal plane.
    Npix_foc = image.shape[0]
    ()
    # generating the pixel grid.
    grid = hcipy.make_uniform_grid([Npix_foc, Npix_foc], [Npix_foc, Npix_foc])
    grid_polar = grid.as_('polar')

    # selecting the first Airy ring
    select = (grid_polar.r > (1.22 + 0.3) * lambda_D_mas / mas_pix) * \
             (grid_polar.r < (2 * 1.22 - 0.3) * lambda_D_mas / mas_pix)
    select = select.reshape(Npix_foc, Npix_foc)
    # Normalizing the image.
    image_norm = image / image.max()

    # Calculating the normalized variations over the reference PSF.
    reference_airy = reference_PSF[select]
    reference_airy /= np.mean(reference_airy)

    # Calculating the normalized variations over the actual PSF.
    airy_data = image_norm[select]
    airy_data /= np.mean(airy_data)

    # Normalizing the variations of the actual PSF with that of the reference PSF.
    airy_data /= reference_airy

    # Calculating the variance and returning it.
    return np.std(airy_data)**2

def quick_strehl_est(inputdata, reference_psf):
    """
    Returns a fast estimate of Strehl using the approximation presented
    in Korkiakoski2014 Eq'n 11-12
    In testing, this agrees within a few percent of the formula presented in
    Bos2021 Eq'n 1

    inputdata - np array or hcipy Field
        the input data -- MUST BE BGD SUBTRACTED
    reference_PSF - np array or hcipy Field
        the reference PSF -- MUST BE BGD SUBTRACTED
    """
    pn = inputdata/np.sum(inputdata)*np.sum(reference_psf)
    ret = np.max(pn)/np.max(reference_psf)
    return np.float(ret)

def calculate_SRA(image, reference_PSF, mas_pix, wavelength, diameter, bg_subtraction=True):
    ''' Calculates the Strehl Ratio Approximation (SRA) of the image.


    For the exact definition, see equation 12 in Bos et al. (2020).

    Parameters
    ----------
    image : HCIPy hcipy.Field object
        The image containing the PSF of the system, centered to the reference PSF.
    reference_PSF : HCIPy hcipy.Field object
        The numerically calculated reference PSF.
    mas_pix : float
        The pixel scale of the detector in milliarcsec per pixel.
    wavelength : float
        The central wavelength of the filter operated in, provided in meter.
    diameter : float
        The projected diameter of the exit pupil in meter.
    bg_subtraction : Boolean
        Used for additional background suppresion to improve the SRA measurement.
    '''
    # Calculating what lambda/D is in milliarcsec.
    lambda_D_rad = wavelength / diameter
    lambda_D_mas = np.degrees(lambda_D_rad) * 3600 * 1000

    # number of pixels along one axis in the focal plane.
    Npix_foc = image.grid.shape[0]

    # diameter of the inner aperture around the core in lambda/D.
    dia_core_LD = 2.44

    # diameter of the outer aperture in lambda/D
    dia_ring_LD = 23

    # diameter of the inner aperture around the core in pixels
    dia_core_pix = dia_core_LD * lambda_D_mas / mas_pix

    # diameter of the inner aperture around the core in pixels
    dia_ring_pix = dia_ring_LD * lambda_D_mas / mas_pix

    # generating the pixel grid.
    grid = hcipy.make_uniform_grid([Npix_foc, Npix_foc], [Npix_foc, Npix_foc])

    # generating the inner and outer masks
    core_mask = hcipy.circular_aperture(dia_core_pix)(grid)
    ring_mask = hcipy.circular_aperture(dia_ring_pix)(grid)

    # the flux in the inner and outer masks for the reference PSF
    sum_core_ref = np.sum(reference_PSF[core_mask==1])
    sum_ring_ref = np.sum(reference_PSF[ring_mask==1])

    # calculating the reference scaling.
    reference = np.sum(reference_PSF[core_mask==1]) / np.sum(reference_PSF[ring_mask==1])

    if bg_subtraction:
        # estimating the background outside of the area used.
        assert len(image[ring_mask==0]) != 0, \
            "Image is too small for effective bgd sub. \
            Plz increase image size if you want to bgd sub"

        bg_est = np.median(image[ring_mask==0])

        # subtracting background estimate
        image -= bg_est

    # the flux in the inner and outer masks for the image
    sum_core = np.sum(image[core_mask==1])
    sum_ring = np.sum(image[ring_mask==1])

    #error_core = np.sqrt(sum_core)
    #rror_ring = np.sqrt(sum_ring)

    # calculating the SRA
    strehl = sum_core / sum_ring / reference

    return strehl

def generate_basis_modes(chosen_mode_basis=None, Nmodes=None, grid_diameter=None, pupil_grid=None):
    """Retrieve the mode basis for the calculation.  Options implemented
       `zernike`, `disk_harmonics`, and `fourier`

    Parameters
    ----------
    chosen_mode_basis : string
        one of `zernike`, `disk_harmonics`, and `fourier`

    Nmodes : int
        the number of basis modes to generate

    grid_diameter : float
        The grid diameter in meters

    pupil_grid : hcipy.field.cartesian_grid.CartesianGrid (possibly other geom)
       The generated pupil grid to project upon

    Returns
    ---------
       an hcipy.mode_basis.mode_basis.hcipy.ModeBasis
    """
    if chosen_mode_basis == 'zernike':
        #make zernike basis ignoring piston/tip/tilt starting at mode 4
        mode_basis = hcipy.make_zernike_basis(Nmodes, grid_diameter, pupil_grid, starting_mode=4)

    elif chosen_mode_basis == 'disk_harmonics':
        mode_basis = hcipy.make_disk_harmonic_basis(pupil_grid, Nmodes, grid_diameter)

    elif chosen_mode_basis == 'fourier':
        # calculating the number of modes along one axis
        Npix_foc_fourier_modes = int(np.sqrt(Nmodes))

        fourier_grid = hcipy.make_uniform_grid([Npix_foc_fourier_modes, Npix_foc_fourier_modes],
                                               [2 * np.pi * Npix_foc_fourier_modes / grid_diameter,
                                                2 * np.pi * Npix_foc_fourier_modes/ grid_diameter])
        mode_basis = hcipy.make_fourier_basis(pupil_grid, fourier_grid)

    elif chosen_mode_basis == 'PTT':
        raise ValueError('Not implemented yet!')
        # need the segments of the aperture

    elif chosen_mode_basis == 'PTT+zernike':
        raise ValueError('Not implemented yet!')
    if len(mode_basis) != Nmodes:
        warnings.warn("Warning: number of modes changed from "+ str(Nmodes)+ \
                      " to " + str(len(mode_basis)) + " due to rounding")
    return mode_basis

def generate_random_phase(rms_wfe=None, mode_basis=None, pupil_grid=None, aperture=None):
    """Generates a random wavefront error over a particular set of modes and aperture
    Parameters
    ----------
    rms_wfe : float [radians]
        The amoutn of WFE to generate, in radians

    mode_basis : hcipy.mode_basis.mode_basis.hcipy.ModeBasis
        The modal basis over which to generate the WFE

    pupil_grid : hcipy.field.cartesian_grid.CartesianGrid
        The pupil grid to use to apply the modal basis

    aperture : hcipy.field.field.hcipy.Field'>
        The aperture over which to normalize the WFE result

    Outputs
    ----------
    applied_phase: hcipy.field.field.hcipy.Field
    """
    #generate random amplitude values.  does not bias towards lower
    #spatial frequencies
    if mode_basis is not None:
        random_amplitudes = np.random.randn(len(mode_basis))
        #dot random amplitdues into mode basis
        random_phase_vals = np.dot(random_amplitudes, mode_basis)
        #create field out of vector
        applied_phase = hcipy.Field(random_phase_vals, pupil_grid)
    else:
        applied_phase = hcipy.Field(np.random.normal(size=pupil_grid.size), pupil_grid)
    #normalize to rms_wfe amount
    applied_phase /= rms(applied_phase, aperture)
    applied_phase *= rms_wfe
    return applied_phase*(aperture>0)

def robust_sigma(in_y, zero=0):
    """
    Calculate a resistant estimate of the dispersion of
    a distribution. For an uncontaminated distribution,
    this is identical to the standard deviation.

    Use the median absolute deviation as the initial
    estimate, then weight points using Tukey Biweight.
    See, for example, Understanding Robust and
    Exploratory Data Analysis, by Hoaglin, Mosteller
    and Tukey, John Wiley and Sons, 1983.

    .. note:: ROBUST_SIGMA routine from IDL ASTROLIB.
    :History:
        * H Freudenreich, STX, 8/90
        * Replace MED call with MEDIAN(/EVEN), W. Landsman, December 2001
        * Converted to Python by P. L. Lim, 11/2009

    Examples
    --------
        result = robust_sigma(in_y, zero=1)

    Parameters
    ----------
    in_y: array_like
        Vector of quantity for which the dispersion is
        to be calculated

    zero: int
        If set, the dispersion is calculated w.r.t. 0.0
        rather than the central value of the vector. If
        Y is a vector of residuals, this should be set.

    Returns
    -------
    out_val: float
        Dispersion value. If failed, returns -1.
    """
    # Flatten array
    y = in_y.reshape(in_y.size, )

    eps = 1.0E-20
    c1 = 0.6745
    c2 = 0.80
    c3 = 6.0
    c4 = 5.0
    c_err = -1.0
    min_points = 3

    if zero:
        y0 = 0.0
    else:
        y0 = np.median(y)

    dy    = y - y0
    del_y = abs( dy )

    # First, the median absolute deviation MAD about the median:

    mad = np.median( del_y ) / c1

    # If the MAD=0, try the MEAN absolute deviation:
    if mad < eps:
        mad = np.mean( del_y ) / c2
    if mad < eps:
        return 0.0

    # Now the biweighted value:
    u  = dy / (c3 * mad)
    uu = u*u
    q  = np.where(uu <= 1.0)
    count = len(q[0])
    if count < min_points:
        #print 'ROBUST_SIGMA: This distribution is TOO WEIRD! Returning', c_err
        return c_err

    numerator = np.sum( (y[q]-y0)**2.0 * (1.0-uu[q])**4.0 )
    n    = y.size
    den1 = np.sum( (1.0-uu[q]) * (1.0-c4*uu[q]) )
    siggma = n * numerator / ( den1 * (den1 - 1.0) )

    if siggma > 0:
        out_val = np.sqrt( siggma )
    else:
        out_val = 0.0

    return out_val
