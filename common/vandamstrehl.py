
#Routines based on code from Marcos Van Dam to reproduce
#the Strehl calculator he uses including method 7 in
#"Is that really your Strehl ratio?", Roberts et al
import numpy as np
from hcipy import make_pupil_grid, evaluate_supersampled
import scipy

def rebin(a,factor,func=None):
    """Rebins an image
    Inputs
    -------
    a - numpy ndarray
        image to rebin
    factor - integer or tuple of ints
        factor to rebin by
    func - function
        function to apply to rebin (default mean)
    Outputs
    -------
    numpy ndarray rebinned

    """
    from numpy.lib.stride_tricks import as_strided
    dim = a.ndim
    if np.isscalar(factor):
        factor = dim*(factor,)
    elif len(factor) != dim:
        raise ValueError('length of factor must be {} (was {})'
                            .format(dim, len(factor)))
    if func is None:
        func = np.mean
    for f in factor:
        if f != int(f):
            raise ValueError('factor must be an int or a tuple of ints '
                                '(got {})'.format(f))

    new_shape = [n//f for n, f in zip(a.shape, factor)]+list(factor)
    new_strides = [s*f for s, f in zip(a.strides, factor)]+list(a.strides)
    aa = as_strided(a, shape=new_shape, strides=new_strides)
    return func(aa, axis=tuple(range(-dim, 0)))

def find_peak(image, xc, yc, boxsize, oversamp=8):
    """
    usage: peak = find_peak(image, xc, yc, boxsize)
    finds the subpixel peak of an image

    image: an image of a point source for which we would like to find the peak
    xc, yc: approximate coordinate of the point source
    boxsize: region in which most of the flux is contained (typically 20)
    oversamp: how many times to oversample the image in the FFT interpolation in order to find the peak

    :return: peak of the oversampled image

    Marcos van Dam, October 2022, translated from IDL code of the same name
    """
    boxhalf = np.ceil(boxsize/2.).astype(int)
    boxsize = 2*boxhalf
    ext = np.array(boxsize*oversamp,dtype=int)

    # need to deconvolve the image by dividing by a sinc in order to "undo" the sampling
    fftsinc = np.zeros(ext)
    fftsinc[0:oversamp]=1.

    sinc = boxsize*np.fft.fft(fftsinc,norm="forward")*np.exp(1j*np.pi*(oversamp-1)*np.roll(np.arange(-ext/2,ext/2),int(ext/2))/ext)
    sinc = sinc.real
    sinc = np.roll(sinc,int(ext/2))
    sinc = sinc[int(ext/2)-int(boxsize/2):int(ext/2)+int(boxsize/2)]
    sinc2d = np.outer(sinc,sinc)

    # define a box around the center of the star
    blx=np.floor(xc-boxhalf).astype(int)
    bly=np.floor(yc-boxhalf).astype(int)

    # make sure that the box is contained by the image
    blx = np.clip(blx,0,image.shape[0]-boxsize)
    bly = np.clip(bly,0,image.shape[1]-boxsize)

    # extract the star
    subim = image[blx:blx+boxsize,bly:bly+boxsize]

    # deconvolve the image by dividing by a sinc in order to "undo" the pixelation
    fftim1 = np.fft.fft2(subim,norm="forward")
    shfftim1 = np.roll(fftim1,(-boxhalf,-boxhalf),axis=(1,0))
    shfftim1 /= sinc2d # deconvolve

    zpshfftim1 = np.zeros((oversamp*boxsize,oversamp*boxsize),dtype='complex64')
    zpshfftim1[0:boxsize,0:boxsize] = shfftim1

    zpfftim1 = np.roll(zpshfftim1,(-boxhalf,-boxhalf),axis=(1,0))
    subimupsamp = np.fft.ifft2(zpfftim1,norm="forward").real

    peak = np.max(subimupsamp)
    return peak

def get_precise_peak_location(im, pos=None, peak_radius=10):
    """Computes a precise image peak location using
       center of mass techniques

    Inputs
    -------
    im - numpy ndarray
        the image containing the psf of interest.
    pos - optional 2-element list or array or tuple
        an initial guess of the psf peak
        defaults to brightest pixel
    peak_radius - integer (default 10)
        the radius about which to use to get a more accurate peak
        in units of pixels

    Outputs
    -------
    xc, yc --> the x and y coordinates of an accurate peak
    """
    if pos is None:
        # find the location of the maximum value of the image
        maxloc = np.where(im == np.amax(im))
        xc = maxloc[0][0]
        yc = maxloc[1][0]

    else:
        xc, yc = pos

    sz = im.shape
    #grid referenced to peak of image
    x,y = np.meshgrid(np.arange(sz[0]) - yc,np.arange(sz[1]) - xc)
    peak_region = np.sqrt(x**2+y**2) < peak_radius
    xc,yc = scipy.ndimage.center_of_mass(im*peak_region)
    return xc, yc

def get_photometry(im, xc, yc, photometry_radius):
    """ Computes a flux in a photometric aperture
    Inputs
    -------
    im - numpy ndarray
        the image containing the psf of interest.
    xc, yc - floats or ints
        an accurate specification of the peak pixel location in x/y
        use get_precise_peak_location to get it
    peak_radius - integer (default 10)
        the radius about which to use to get a more accurate peak
        in units of pixels

    Outputs
    -------
    xc, yc --> the x and y coordinates of an accurate peak
    """
    sz = im.shape
    #grid referenced to peak of image
    x,y = np.meshgrid(np.arange(sz[0]) - yc,np.arange(sz[1]) - xc)
    phot_region = np.sqrt(x**2+y**2) < photometry_radius
    #sum up all the flux
    flux = np.sum(im*phot_region)
    return flux

def strehl(im, psf0, pos=None,
           photometry_radius=20,
           peak_radius=10):
    """calculates the Strehl ratio of a PSF.

    Inputs
    --------
    im - 2d numpy array
        The image of the PSF under test
        MUST BE BGD SUBTRACTED
    psf0 - 2d numpy array
        The image of a perfect PSF of the same optical system
        Does not need to be normalized
        MUST BE BGD SUBTRACTED
    pos - 2-element iterable
        the x and y position (in pixels) of the peak, as a guess
        None is also acceptable, then it will guess the maximally
        bright point
    photometry_radius - integer
        the radius (in pixels) to calculate the total flux
    peak_radius - integer
        the radius (in pixels) about which to try and find the peak

    Outputs
    ---------
    strehl - float
        the strehl ratio
    """
    #get the precise peak of the image first
    xc, yc = get_precise_peak_location(im, pos=pos,
                                       peak_radius=peak_radius)
    #Calculate the PSF photometry--MUST BE BGD SUBTRACTED!
    flux_im = get_photometry(im, xc, yc, photometry_radius)
    peak_im = find_peak(im, xc, yc, peak_radius)

    # now repeat for the diffraction-limited PSF
    psf_xc, psf_yc = get_precise_peak_location(psf0, pos=None,
                                               peak_radius=10)
    flux_psf = get_photometry(psf0, psf_xc, psf_yc,
                              photometry_radius)
    peak_psf = find_peak(psf0, psf_xc, psf_yc,
                              peak_radius)
    #now compute the strehl
    strehl = (peak_im/flux_im)/(peak_psf/flux_psf)
    return strehl
