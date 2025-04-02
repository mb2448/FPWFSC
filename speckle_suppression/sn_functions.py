import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter


def flip_array_about_point(arr, point_x, point_y):
    """
    Flip a 2D array about a specified point in both x and y directions.
    
    Parameters:
    arr (numpy.ndarray): 2D input array to be flipped
    point_x (float): x-coordinate of the point to flip about
    point_y (float): y-coordinate of the point to flip about
    
    Returns:
    numpy.ndarray: Flipped array
    """
    # Get array dimensions
    height, width = arr.shape
    
    # Create coordinate meshgrid
    y, x = np.indices((height, width))
    
    # Calculate new coordinates after flipping about the point
    new_x = 2 * point_x - x
    new_y = 2 * point_y - y
    
    # Create output array with same shape as input
    flipped = np.zeros_like(arr)
    
    # Map values from original array to flipped positions
    # Need to handle edge cases where new coordinates are outside the array
    valid_indices = (new_x >= 0) & (new_x < width) & (new_y >= 0) & (new_y < height)
    
    # For valid indices, copy values from original array to flipped array
    y_valid, x_valid = y[valid_indices], x[valid_indices]
    new_y_valid, new_x_valid = new_y[valid_indices].astype(int), new_x[valid_indices].astype(int)
    
    flipped[y_valid, x_valid] = arr[new_y_valid, new_x_valid]
    
    return flipped



def annulus(image, cx, cy, r1, r2):
    outer = circle(image, cx, cy, r2)
    inner = circle(image, cx, cy, r1)
    return ( outer-inner)

def circle(image, cx, cy, rad):
    zeroim = np.zeros(image.shape, dtype = np.int)
    for x in range(int(cx-rad), int(cx+rad+1)):
        for y in range(int(cy-rad), int(cy+rad+1) ):
            #print xs, ys
            dx = cx-x
            dy = cy -y
            if(dx*dx+dy*dy <= rad*rad):
                zeroim[y,x] = 1
    return zeroim

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
   >>> result = robust_sigma(in_y, zero=1)

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

def contrastcurve_simple(image, cx=None, cy = None,
                         sigmalevel = 1, robust=True,
                         region =None, maxrad = None):
    """image - your bgd-subtracted image
       cx  - image center [pix]
       cy  - image center [pix]
       robust - use robust sigma
       region - control region
       maxrad - max radius to calculate [pix]"""
    if cx is None:
        cx = image.shape[0]//2
    if cy is None:
        cy = image.shape[0]//2

    if maxrad is None:
        maxpix = image.shape[0]//2
    else:
        maxpix = maxrad
    pixrad = np.arange(maxpix)
    clevel = pixrad*0.0
    if region is None:
        region = np.ones(image.shape)
    for idx, r in enumerate(pixrad):
        annulusmask = annulus(image, cx, cy, r, r+1)
        if robust:
            sigma = robust_sigma(image[np.where(np.logical_and(annulusmask, region))].ravel())
        else:
            sigma = np.std(image[np.where(np.logical_and(annulusmask, region))].ravel())

        clevel[idx]=sigmalevel*(sigma)
    return (pixrad, clevel)


def twoD_Gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """Returns a two-d gaussian function
    coordinate variables:
    x - the x coordinate.  may be an array
    y - the y coordinate.  may be an array

    function variables
    amplitude - the peak amplitude of the gaussian
    xo - the x centroid of the gaussian
    yo - the y centroid of the gaussian
    sigma_x - the std dev of the gaussian in x
    sigma_y - the std dev of the gaussian in y
    theta - the rotation angle (in radians) - Note--degenerate with sigmaxy when off by 180
    offset - the background level

    Returns
    a scalar or vector or array of z-values at each x, y
    """
    #x, y = xytuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( -(a*((x-xo)**2)
                                    + 2*b*(x-xo)*(y-yo)
                                    + c*((y-yo)**2)))
    return g

def twoD_Gaussian_fitfunc(xytuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    #passes a single parameter for x, y
    #ravels hte output
    return twoD_Gaussian(xytuple[0], xytuple[1], amplitude, xo, yo, sigma_x, sigma_y, theta, offset).ravel()

def fitgaussian(image, x=None, y=None):
    """Returns a fit for a gaussian function given an image input"""
    if (x is None) or (y is None):
        x, y = np.meshgrid( np.arange(image.shape[1]),
                            np.arange(image.shape[0]))
    xindguess= np.argwhere(np.nansum(image, axis=0)==np.max(np.nansum(image, axis=0)))[0][0]
    yindguess= np.argwhere(np.nansum(image, axis=1)==np.max(np.nansum(image, axis=1)))[0][0]

    xoguess = x[yindguess, xindguess]
    yoguess = y[yindguess, xindguess]
    offsetguess = np.nanmean(image)
    amplitudeguess = np.nanmax(image)-offsetguess
    initguess = (amplitudeguess,
                 xoguess,
                 yoguess,
                 1.0,
                 1.0,
                 0.0,
                 offsetguess)
    inds = np.isfinite(image)
    input_image = image[inds]
    input_x = x[inds]
    input_y = y[inds]
    popt, pcov = opt.curve_fit(twoD_Gaussian_fitfunc, (input_x, input_y),
                   input_image.ravel(),
                   p0 = initguess, maxfev = 100000000)
    return popt

def image_centroid_gaussian(image, x=None, y=None):
    """Returns the coordinates of the center of a gaussian blob in the image"""
    popt = fitgaussian(image, x=x, y=y)
    return popt[1], popt[2]

# =============================================================================
def get_spot_centroid(image, window = 20, guess_spot=None):
    ''' -----------------------------------------------------------------------
    Measures the centroid of a spot accurately by fitting a guassian on a
    subimage centered on the spot.

    image - a numpy array containing the image
    window - the size of the subimage in pixels
    guess_spot - a tuple containing the (x,y) coordinates of the spot
    ----------------------------------------------------------------------- '''
    # Measure acurately the position of each satellite
    # Generate a sub image centered on the xy position
    y0    = int(round(guess_spot[0]))
    x0    = int(round(guess_spot[1]))
    hw    = int(round(window/2.))
    subim =  image[x0-hw:x0+hw,y0-hw:y0+hw]
    # Fit a gaussian on the subimage roughtly centered on the speckle.
    popt  = image_centroid_gaussian(subim)
    # Extract center position (x,y) of the gaussian fitted on the speckles.
    xcen  = round(y0-hw+popt[0],3)
    ycen  = round(x0-hw+popt[1],3)
    # Add the position computed to the list of positions.
    return xcen, ycen


def create_annular_wedge(image, xcen, ycen, rad1, rad2, theta1, theta2):
    """
    Create an annular wedge mask and apply it to an image.

    Parameters:
    -----------
    image : 2D numpy array
        The input image
    xcen, ycen : int
        Central pixel coordinates
    rad1, rad2 : float
        Inner and outer radii of the annular wedge in pixels
    theta1, theta2 : float
        Start and end angles of the wedge in degrees (measured counterclockwise from the positive x-axis)
        Can use negative angles (e.g., -90 to 90 for a half-circle on the right side)

    Returns:
    --------
    masked_image : 2D numpy array
        Image with only the annular wedge region visible
    mask : 2D numpy array
        Binary mask of the annular wedge region
    """
    # Create coordinate grids
    y, x = np.indices(image.shape)

    # Calculate distance from center for each pixel
    r = np.sqrt((x - xcen)**2 + (y - ycen)**2)

    # Calculate angle for each pixel (in degrees)
    # arctan2 returns angles in radians in range [-π, π], so convert to degrees
    theta = np.rad2deg(np.arctan2(y - ycen, x - xcen))

    # Create the angle mask based on the input angles
    # No need to normalize angles - we can directly compare with the raw arctan2 output
    if theta1 <= theta2:
        # Simple case: theta1 to theta2 without crossing -180/180 boundary
        angle_mask = (theta >= theta1) & (theta <= theta2)
    else:
        # Case where the wedge crosses the -180/180 boundary
        angle_mask = (theta >= theta1) | (theta <= theta2)
    
    # Create the annular wedge mask
    mask = (r >= rad1) & (r <= rad2) & angle_mask

    # Apply mask to the image
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]

    # MC modif
    """
    # Create a soft radial transition at the edges
    radial_inner = np.exp(-((r - rad1) / sigma) ** 2) * (r < rad1)
    radial_outer = np.exp(-((r - rad2) / sigma) ** 2) * (r > rad2)

    # Convert angle mask from binary to soft transition
    angle_blur = gaussian_filter(angle_mask.astype(float), sigma=sigma)

    # Create a smoothed mask
    smoothed_mask =  mask.astype(float)
    smoothed_mask += radial_inner + radial_outer  # Apply edge smoothing
    smoothed_mask *= angle_blur  # Apply angular smoothing

    # Clip values to [0,1]
    smoothed_mask = np.clip(smoothed_mask, 0, 1)
    smoothed_mask = smoothed_mask.astype(bool)

    # Apply the smoothed mask to the image
    masked_image = smoothed_mask * image
    """
    return mask
