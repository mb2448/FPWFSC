import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

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
    
    return mask