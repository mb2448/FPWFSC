############################### Import Library ################################

## Math Library
import numpy as np
# Import scipy library optimize (mainly use for function fits)
import scipy.optimize as opt 
import pdb


#import scipy.special as special

###############################################################################

# =============================================================================
def fitgaussian(im, x=None, y=None):
    ''' -----------------------------------------------------------------------    
    Returns a fit for a gaussian function given an image input"""
    ----------------------------------------------------------------------- '''
    # Generates the x and y parameter based on the shape of the image if not
    # porvided by the user
    if (x is None) or (y is None):
        list_x = np.arange(im.shape[1])
        list_y = np.arange(im.shape[0]) 
        x, y = np.meshgrid(list_x,list_y)
    
    xindguess = np.argwhere(np.nansum(im, axis=0)==np.max(np.nansum(im, axis=0)))[0][0]
    yindguess = np.argwhere(np.nansum(im, axis=1)==np.max(np.nansum(im, axis=1)))[0][0] 
    
    xoguess = x[yindguess, xindguess]
    yoguess = y[yindguess, xindguess]
    offsetguess = np.nanmean(im)
    amplitudeguess = np.nanmax(im)-offsetguess
    initguess = (amplitudeguess, xoguess, yoguess, 1.0, 1.0, 0.0, offsetguess)
    inds = np.isfinite(im)
    input_image = im[inds]
    input_x = x[inds]
    input_y = y[inds]
    popt, pcov = opt.curve_fit(twoD_Gaussian_fitfunc, (input_x, input_y), 
                   input_image.ravel(),
                   p0 = initguess, maxfev = 100000000)
    return popt




def rotateXY(xvals, yvals, thetadeg = 0):
    theta = np.pi/180.0*thetadeg
    return (np.cos(theta)*xvals- np.sin(theta)*yvals, 
            np.sin(theta)*xvals+ np.cos(theta)*yvals)


def point_in_poly(x, y, poly):
    #poly is a list of x, y pairs
    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def points_in_poly(xvals, yvals, polyvertices):
    #use meshgrid before
    ans=np.zeros(np.shape(xvals))
    for i in range(np.shape(xvals)[0]):
        for j in range(np.shape(xvals)[1]):
            if point_in_poly(xvals[i, j], yvals[i,j], polyvertices):
                ans[i,j] = 1
    return ans


#-----Robust Mean--------------
def robust_mean(x):
    y = x.flatten()
    n = len(y)
    y.sort()
    ind_qt1 = round((n+1)/4.)
    ind_qt3 = round((n+1)*3/4.)
    IQR = y[ind_qt3]- y[ind_qt1]
    lowFense = y[ind_qt1] - 1.5*IQR
    highFense = y[ind_qt3] + 1.5*IQR
    ok = (y>lowFense)*(y<highFense)
    yy=y[ok]
    return yy.mean(dtype='double')


#-------Robust Standard Deviation---

def robust_std(x):
    y = x.flatten()
    n = len(y)
    y.sort()
    ind_qt1 = round((n+1)/4.)
    ind_qt3 = round((n+1)*3/4.)
    IQR = y[ind_qt3]- y[ind_qt1]
    lowFense = y[ind_qt1] - 1.5*IQR
    highFense = y[ind_qt3] + 1.5*IQR
    ok = (y>lowFense)*(y<highFense)
    yy=y[ok]
    return yy.std(dtype='double')

#-------Robust Standard Deviation, version 2---
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
#-------Robust variance---

def robust_var(x):
    y = x.flatten()
    n = len(y)
    y.sort()
    ind_qt1 = round((n+1)/4.)
    ind_qt3 = round((n+1)*3/4.)
    IQR = y[ind_qt3]- y[ind_qt1]
    lowFense = y[ind_qt1] - 1.5*IQR
    highFense = y[ind_qt3] + 1.5*IQR
    ok = (y>lowFense)*(y<highFense)
    yy=y[ok]
    return yy.var(dtype='double')


def ideal2dpsf(xs, ys,  xc, yc, 
               pix = 25, lambdaoverd=90.7, aoverA=.32,
               fudgefactor = 2.15, amp = 1):
    scalefact = lambdaoverd/pix
    v = np.hypot(xs-xc, ys-yc)*np.pi/scalefact
        
    a= (2*special.jn(1, v)/v)
    b= -aoverA**2*fudgefactor*2*special.jn(1,v*aoverA)/(aoverA*v)
    retval = (a+b)**2
    retval[np.isnan(retval)]=(2*0.5 - aoverA**2*2*0.5)**2
    return amp*retval/np.max(retval)


def gausspsf2d(npix, fwhm, normalize=True): 
    """ 
    Parameters 
    ---------- 
    npix: int 
        Number of pixels for each dimension. 
        Just one number to make all sizes equal. 

    fwhm: float 
        FWHM (pixels) in each dimension. 
        Single number to make all the same. 

    normalize: bool, optional 
        Normalized so total PSF is 1. 

    Returns 
    ------- 
    psf: array_like 
        Gaussian point spread function. 
    """ 

    # Initialize PSF params 
    cntrd = (npix - 1.0) * 0.5 
    st_dev = 0.5 * fwhm / np.sqrt( 2.0 * np.log(2) ) 

    # Make PSF 
    x, y = np.indices([npix,npix]) - (npix-1)*0.5
    psf = np.exp( -0.5 * ((x**2 + y**2)/st_dev**2) )
    # Normalize 
    if normalize: psf /= psf.sum() 

    return psf

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


# =============================================================================
def image_centroid_gaussian(image, x=None, y=None):
    ''' -----------------------------------------------------------------------
    Returns the coordinates of the center of a gaussian blob in the image.
    ----------------------------------------------------------------------- '''
    # Basically call an other function to do the job
    popt = fitgaussian(image, x=x, y=y)
    return popt[1], popt[2]


