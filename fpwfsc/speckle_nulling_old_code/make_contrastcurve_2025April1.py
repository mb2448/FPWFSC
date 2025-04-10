import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

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
                         fwhm = 1, sigmalevel = 1, robust=True,
                         region =None, maxrad = None):
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
        annulusmask = annulus(image, cx, cy, r, r+fwhm)
        if robust:
            sigma = robust_sigma(image[np.where(np.logical_and(annulusmask, region))].ravel())
        else:
            sigma = np.std(image[np.where(np.logical_and(annulusmask, region))].ravel())

        clevel[idx]=sigmalevel*(sigma)
        print(str(idx) + ' '+str( clevel[idx]))
    return (pixrad, clevel)

def generate_curveimage(image, pixrad, clevel, cx=None, cy = None):
    """Returns an image where the intensity at each point corresponds
        to the contrast level you pass as clevel"""
    if cx is None:
        cx = image.shape[0]//2
    if cy is None:
        cy = image.shape[0]//2
    outim = np.ones(image.shape)
    for idx, r in enumerate(pixrad):
        annulusmask = annulus(image, cx, cy, r, r+1)
        outim[np.where(annulusmask)]=clevel[idx]
    return outim


def contrastcurve(image, psf, cx = None, cy = None,
                  plsc = .025, fwhm = 3.4, kernel = None,
                  sigmalevel=5, conv=False, robust = False,
                  max_fov=6, filtermultfactor = 36.0):
    """compute the contrast curve from a reduced image
       and a psf image.
        image = image to compute curve for
        cx, cy = center x and center y of image (pixels)
        psf   = psf image for contrast zeropoint
        sigmalevel = 5
        conv = T/F whether to perform a gaussian convolution
                    on the image
        fwhm = fwhm of gaussian to convolve AND used to compute
               annular width in contrast curve
        plsc = platescale (arc-seconds/pixel)
        max_fov = maximum field of view in arc seconds
        robust  = use a robust sigma measure
        multfactor = any multiplication factor resulting from
                     different filters or grisms
    """

    if (cx is None) or (cy is None):
        cx, cy = pre.get_spot_locations(image, comment='click on the center pixel')[0]
    if conv:
        print( "Convolving PSF with kernel")
        convim = pro.matchfilter(image, kernel)
        print( "Fitting PSF")
        psfamp=pre.quick2dgaussfit(pro.matchfilter(psf, kernel), xy = [cx, cy])[0]
    else:
        convim = image
        psfamp = pre.quick2dgaussfit(psf, xy = [cx, cy])[0]

    maxpix = min(int(max_fov/plsc), image.shape)
    pixrad = np.arange(maxpix)
    clevel = np.arange(maxpix)*0.0

    #plt.ion()
    #fig, ax0 = plt.subplots(ncols=1, figsize = (8, 8))
    #ax=plt.imshow(convim, interpolation='nearest',origin='lower')
    #plt.show()
    time.sleep(1)
    for idx, r in enumerate(pixrad):
        annulusmask = pro.annulus(convim, cx, cy, r, r+fwhm)
        #sigma = rs.robust_sigma(convim[np.where(annulusmask)].ravel())
        if idx <1:
            sigma = None
            clevel[idx]==None
            continue
        if robust is True:
            sigma = plm.robust_std(convim[np.where(annulusmask)].ravel())
        else:
            sigma = np.std(convim[np.where(annulusmask)].ravel())

        clevel[idx]=sigmalevel*(sigma)/psfamp/filtermultfactor
        printout = [str(x) + ', ' for x in [r,pixrad[idx]*plsc,clevel[idx] ]]
        print( ''.join(printout))
        #ax.set_data(convim*annulusmask)
        #plt.draw()
    #plt.close()
    #plt.ioff()
    return (pixrad*plsc, clevel)

def rawcontrastcurve(image, psf=None, cx = None, cy = None,
                  fwhm = 3.4, plsc = .025,
                  sigmalevel=5, gaussfilt=False,robust = False,
                  max_fov=6, multfactor = 36.0, rawmultfactor = 1.0):
    if cx is None or cy is None:
        #cx, cy = pre.quickcentroid(image)
        cx, cy = 256, 256
    maxpix = min(int(max_fov/plsc), image.shape)
    if gaussfilt:
        kernel = plm.gausspsf2d(10, fwhm)
        convim = sciim.filters.convolve(image, kernel)
    else:
        convim = image

    if psf is not None:
        psfamp=pre.quick2dgaussfit(psf)[0]

    pixrad = np.arange(maxpix)
    clevel = np.arange(maxpix)*0.0

    for idx, r in enumerate(pixrad):
        annulusmask = pro.annulus(convim, cx, cy, r, r+fwhm)
        if idx <1:
            sigma = None
            clevel[idx]==None
            continue
        if robust == True:
            sigma = plm.robust_std(convim[np.where(annulusmask)].ravel())
        else:
            sigma = np.std(convim[np.where(annulusmask)].ravel())
        if psf is not None:
            clevel[idx]=sigmalevel*(sigma)/psfamp/multfactor
        else:
            clevel[idx]=sigmalevel*sigma/rawmultfactor
        print( str(idx)+", "+ str(pixrad[idx]*plsc)+", "+ str(clevel[idx]))

    return (pixrad*plsc, clevel)

if __name__ == "__main__":
    datafile = "/Users/mbottom/Desktop/projects/HSF_exoplanet_imaging/fast_and_furious_wfs/code/FPWFSC/speckle_suppression/speckle_nulling_old_code/ref_img_Halfdark_ND1_5ms.fits"
    data = fits.open(datafile)[0].data


    controlregionfile = "/Users/mbottom/Desktop/projects/HSF_exoplanet_imaging/fast_and_furious_wfs/code/FPWFSC/speckle_suppression/speckle_nulling_old_code/controlregion.fits"

    controlregion = fits.open(controlregionfile)[0].data

    #pixrad, clevel = contrastcurve_simple(data, cx = 252, cy=153, region=controlregion, maxrad = 40)

    datafiles = ["/Users/mbottom/Desktop/projects/HSF_exoplanet_imaging/fast_and_furious_wfs/code/FPWFSC/speckle_suppression/speckle_nulling_old_code/ref_img_Halfdark_ND1_5ms.fits", "/Users/mbottom/Desktop/projects/HSF_exoplanet_imaging/fast_and_furious_wfs/code/FPWFSC/speckle_suppression/speckle_nulling_old_code/SAN_iter0_Halfdark_ND1_5ms.fits", "/Users/mbottom/Desktop/projects/HSF_exoplanet_imaging/fast_and_furious_wfs/code/FPWFSC/speckle_suppression/speckle_nulling_old_code/SAN_iter1_Halfdark_ND1_5ms.fits"]
    for i, datafile in enumerate(datafiles):

        data = fits.open(datafile)[0].data
        pixrad, clevel = contrastcurve_simple(data, cx = 252, cy=153, region=controlregion, maxrad = 40)

        plt.plot(pixrad, clevel, label=f'Iteration {i}')
    plt.xlabel('x pixel')
    plt.ylabel('contrast 1-sig raw')
    plt.legend()
    plt.show()
