# -*- coding: utf-8 -*-

"""
Contains the functions for computing tip-tilt estimates based on the QACITS
method. 
"""

__author__ = 'E. Huby @ ULg'
__all__ = ['qacits_subimage', 
           'qacits_delta_i',
           'qacits_circle_mask', 
           'qacits_estimate_tiptilt', 
           'qacits_get_estimator', 
           'qacits_get_psf_flux', 
           'qacits_subtract_sky']


import numpy as np
#import photutils
import matplotlib.pyplot as plt


def qacits_subimage(image, cx, cy, quadrant_width, full_output=False):                          
    """ This routine creates a sub-image from a given image centered on the 
    coordinates (cx,cy). This sub-image is not necessarily a square, and 
    the exact width in each direction depends on the parity of 2*cxy (cxy 
    being cx or cy, and can be a fractional number). There are two different 
    cases: 
        - the center falls between two pixels (2*cxy is odd), then the width 
        is even (2*quadrant_width)
        - the center falls in the middle of one pixel (2*cxy is even), then 
        the width is odd (2*quadrant_width+1)
    
    Parameters
    ----------
    image : array_like 
        input image for which the sub-image needs to be created.
    cx : float
        x position of the sub-image center [pix], can be fractional.
    cy : float
        y position of the sub-image center [pix], can be fractional.
    full_output : {True, False}, bool optional
        If True, the coordinates of the lower left pixels are returned.
    
    Returns
    -------
    subimage : array_like
        Subimage.  
    """
    
    ny, nx = image.shape
    
    cx2 = np.round(cx*2.)/2.
    cy2 = np.round(cy*2.)/2.
    
    x1 = np.ceil(cx2 - quadrant_width)
    if x1 < 0 or x1 > nx :
        x1 = 0.
        
    x2 = np.floor(cx2 + quadrant_width) + 1.
    if x2 < 0 or x2 > nx :
        x2 = nx 
    
    y1 = np.ceil(cy2 - quadrant_width)
    if y1 < 0 or y1 > ny :
        y1 = 0.
        
    y2 = np.floor(cy2 + quadrant_width) + 1.
    if y2 < 0 or y2 > ny :
        y2 = ny 
    
    subimage = image[y1:y2, x1:x2].copy()
    
    if full_output:
        return subimage, x1, y1
    else:
        return subimage

def qacits_estimate_tiptilt(img, inner_rad_pix, outer_rad_pix, flux_psf,
                            cx = None, cy = None, 
                            instrument = 'nirc2', 
                            small_tt_model = None, large_tt_model = None, 
                            tt_regime_lim = None):
    """ This routine estimates the tip-tilt affecting a VVC image based on the 
    QACITS method. Two regimes can be defined for small and large tip-tilt values.
    In each case, the QACITS model is defined either by default values 
    depending on the instrument, or manually by the user. In the latter case,
    parameters for the small and large tip-tilt regimes must be defined, as
    well as the threshold between the two regimes.
    
    Parameters
    ----------
    img : array_like
        Input 2D array for a single image or 3D array, for a cube.
    inner_rad_pix : float
        Radius of the inner region used for the inner QACITS estimator.
    outer_rad_pix : float
        Radius of the outer region used for the outer QACITS estimator.
    flux_psf : float
        Flux of the off-axis PSF measured in the area of radius outer_rad_pix.
        This Flux must be estimated for the same integration time as the
        science images.
    cx : float, optional
        x position of the sub-image center [pix], can be fractional.
        If not specified, the center is defined at the center of the image.
    cy : float, optional
        y position of the sub-image center [pix], can be fractional.
        If not specified, the center is defined at the center of the image.
    instrument : {'nirc2', 'visir'}, optional
        Name of the instrument that has been used to acquire the data.
    small_tt_regime : list (string1, string2, coefficient)
        Model parameters for the small tip-tilt regime.
        The 1st string of the list, {'inner', 'outer'}, indicates which region
        of the image has to be used.
        The 2nd string of the list, {'linear', 'cubic'}, indicates which kind
        of approximation has to be used.
        The last element of the list is a float and corresponds to the 
        coefficient of the linear/cubic model.
    large_tt_model : list (string1, string2, coefficient)
        Model parameters for the large tip-tilt regime. The format of the list 
        is the same as for small_tt_model.
    tt_regime_lim : float
        The threshold between small and large tip-tilt regime, in unit of 
        lambda/D. The parameters of the small tip-tilt regime will be used at
        first to get a first estimation of the tip-tilt to decide the regime.
    
    Returns
    -------
    final_est : array_like
        Final QACITS estimates corresponding to the input images, given in 
        lambda/D. Dimensions are (2) for a single image, and (n,2) for a cube 
        of n images.
    """
    
    img_sh = img.shape
    nn = len(img_sh)
 
    if nn == 2 :
        # Single image
        n_img = 1
        ny, nx = img_sh
        img = img[None,:]
    elif nn == 3 :
        # Cube of images
        n_img = img_sh[0]
        ny, nx = img_sh[1:3]
    
    if cx == None :
        cx = (nx-1.) / 2.
    if cy == None :
        cy = (ny-1) / 2.
    
    # Model parameters defined by the instrument
    if small_tt_model != None and large_tt_model != None :
        print( 'User defined model parameters will be used')
    elif instrument == 'nirc2':
        print( '##### NIRC2 INSTRUMENT #####')
        small_tt_model = ('outer', 'linear', 0.085)
        large_tt_model = small_tt_model
        tt_regime_lim= 0.
    elif instrument == 'visir':
        print( '##### VISIR INSTRUMENT #####')
        small_tt_model = ('outer', 'linear', 0.03)
        large_tt_model = ('inner', 'cubic', 0.61)
        tt_regime_lim = 0.3
    else:
        raise ValueError('Unknown instrument name OR undefined model parameters.')
    
    final_est = np.zeros((n_img, 2))    
    
    for i in range(n_img):
        image = img[i]
        # compute the estimate using the small tiptilt regime
        small_tt_est = qacits_get_estimator(image, small_tt_model, inner_rad_pix, 
                                            outer_rad_pix, flux_psf, cx=cx, cy=cy)
        small_amp = np.sqrt(small_tt_est[0]**2. + small_tt_est[1]**2.)
        
        # compute the estimate using the large tiptilt regime
        large_tt_est = qacits_get_estimator(image, large_tt_model, inner_rad_pix,
                                            outer_rad_pix, flux_psf, cx=cx, cy=cy)
        large_amp = np.sqrt(large_tt_est[0]**2. + large_tt_est[1]**2.)
        
        tt_amp = np.max([small_amp, large_amp])
        
        if tt_amp < tt_regime_lim :
            # SMALL tip-tilt regime        
            final_est[i] = small_tt_est
        else :
            # LARGE tip-tilt regime
            final_est[i] = large_tt_est

    if n_img == 1:
        return final_est[0]
    else :
        return final_est
    
        

def qacits_get_estimator(image, model_params, inner_rad_pix, outer_rad_pix, 
                         flux_psf, cx=None, cy=None):
    """ This routines estimates the tip-tilt of a VVC image for a given case 
    of the QACITS method.
    
    Parameters
    ----------
    image : array_like
        Input 2D array.
    model_params  : list (string1, string2, coefficient)
        Model parameters for the QACITS model.
        The 1st string of the list, {'inner', 'outer'}, indicates which region
        of the image has to be used.
        The 2nd string of the list, {'linear', 'cubic'}, indicates which kind
        of approximation has to be used.
        The last element of the list is a float and corresponds to the 
        coefficient of the linear/cubic model.
    inner_rad_pix : float
        Radius of the inner region used for the inner QACITS estimator.
    outer_rad_pix : float
        Radius of the outer region used for the outer QACITS estimator.
    flux_psf : float
        Flux of the off-axis PSF measured in the area of radius outer_rad_pix.
        This Flux must be estimated for the same integration time as the
        science images.
    cx : float, optional
        x position of the sub-image center [pix], can be fractional.
        If not specified, the center is defined at the center of the image.
    cy : float, optional
        y position of the sub-image center [pix], can be fractional.
        If not specified, the center is defined at the center of the image.
    
    Returns
    -------
    qacits_est : array_like
        2D element array corresponding to the QACITS estimates in x and y, 
        respectively, and given in lambda/D.        
    """

    ny, nx = image.shape    
    
    if cx == None :
        cx = (nx-1.) / 2.
    if cy == None :
        cy = (ny-1) / 2.
        
    # define the region
    if model_params[0] == 'inner':
        region_mask = qacits_circle_mask(image, inner_rad_pix, cx=cx, cy=cy)
    else:
        region_mask = qacits_circle_mask(image, outer_rad_pix, cx=cx, cy=cy) - qacits_circle_mask(image, inner_rad_pix, cx=cx, cy=cy)
    
    # compute delta_i
    ix, iy = qacits_delta_i(image*region_mask, cx=cx, cy=cy) 
    # normalization
    ix = ix / flux_psf
    iy = iy / flux_psf
    
    # amplitude and orientation angle
    delta_i = np.sqrt(ix**2. + iy**2.)
    theta   = np.arctan2(iy, ix)

    if model_params[1] == 'linear':
        lin_slope = model_params[2]
        final_est = delta_i / lin_slope
    else:
        cub_coeff = model_params[2]
        final_est = (delta_i / cub_coeff)**(1./3.)
    
    qacits_est = [final_est * np.cos(theta), final_est*np.sin(theta)]
    
    return qacits_est

def qacits_delta_i(image, cx=None, cy=None):
    """ Computes the differential intensities along the x and y axes.
    
    Parameters
    ----------
    image : array_like
        Input 2D array.
    cx : float, optional
        x position of the sub-image center [pix], can be fractional.
        If not specified, the center is defined at the center of the image.
    cy : float, optional
        y position of the sub-image center [pix], can be fractional.
        If not specified, the center is defined at the center of the image.
    
    Returns
    -------
    delta_i : array_like
        2D element containing the differential intensities measured along the
        x and y axes.
    """
    
    ny, nx = image.shape    
    
    if cx == None :
        cx = (nx-1.) / 2.
    if cy == None :
        cy = (ny-1) / 2.
    
    Sx = np.cumsum(np.sum(image, axis=0))
    Sy = np.cumsum(np.sum(image, axis=1))
    
    Ix = Sx[-1] - 2. * np.interp(cx, np.arange(nx)+.5, Sx)
    Iy = Sy[-1] - 2. * np.interp(cy, np.arange(ny)+.5, Sy)
    
    delta_i = [Ix, Iy]
    
    return delta_i

def qacits_circle_mask(image, radius, cx=None, cy=None):
    """ Creates a circular mask of same dimensions of input image, and defined
    by a radius and center coordinates. Values are 1 inside the circle, 0 outside.
    
    Parameters
    ----------
    image : array_like
        Input 2D array.
    radius : float
        Radius of the mask in pixels.
    cx : float, optional
        x position of the sub-image center [pix], can be fractional.
        If not specified, the center is defined at the center of the image.
    cy : float, optional
        y position of the sub-image center [pix], can be fractional.
        If not specified, the center is defined at the center of the image.
    
    Returns
    -------
    circle_mask : array_like
        2D array containing 1 and 0 values only.
    """
    ny, nx = image.shape    
    
    if cx == None :
        cx = (nx-1.) / 2.
    if cy == None :
        cy = (ny-1) / 2.
        
    cx2 = np.round(cx*2.)/2.
    cy2 = np.round(cy*2.)/2.
    
    gridy, gridx = np.indices(image.shape)
    gridx = gridx - cx2
    gridy = gridy - cy2
    gridr = np.sqrt(gridx**2. + gridy**2.)
    
    circle_mask = gridr < radius
    
    return circle_mask

def qacits_get_psf_flux(psf, radius, cx=None, cy=None, t_int=1.):
    """ Estimates the flux in the psf input image by integrating the pixel 
    counts in a circular area. Integration time of the image can also be provided
    in order to normalize the final flux estimate.
    
    Parameters
    ----------
    psf : array_like
        Input 2D array.
    radius : float
        Radius of the mask in pixels.
    cx : float, optional
        x position of the sub-image center [pix], can be fractional.
        If not specified, the center is defined at the center of the image.
    cy : float, optional
        y position of the sub-image center [pix], can be fractional.
        If not specified, the center is defined at the center of the image.
    t_int : float, optional
        Integration time of the psf image.
    
    Returns
    -------
    psf_flux : float
        Flux of the psf integrated in the circular area and normalized by its 
        integration time if provided.
    """
    
    ny, nx = psf.shape
    
    if cx == None :
        cx = (nx-1.) / 2.
    if cy == None :
        cy = (ny-1) / 2.

    if cx == None :
        cx = (nx-1.)/2.
    if cy == None :
        cy = (ny-1.)/2.
    
    aper = photutils.CircularAperture((cx, cy), radius)
    obj_flux = photutils.aperture_photometry(psf, aper, method='exact')
    psf_flux = np.array(obj_flux['aperture_sum'])  
    
    return psf_flux[0] / t_int
    
def qacits_subtract_sky(img, sky, null_median = True, cx=None, cy=None, radius_in=None, 
                        radius_out=None, verbose=False, vmin=-500, vmax=500):
    ny, nx = img.shape
    ny2, nx2 = sky.shape

    if nx != nx2 or ny != ny2:
        raise TypeError('Dimensions of img and sky are different.')
    
    if cx == None :
        cx = (nx-1.) / 2.
    if cy == None :
        cy = (ny-1) / 2.

    if radius_in == None :
        radius_in = np.min(nx,ny)*1./3.
    if radius_out == None :
        radius_out = np.min(nx, ny)*2./3.
    
    annulus_mask = qacits_circle_mask(img, radius_out, cx=cx, cy=cy) - qacits_circle_mask(img, radius_in, cx=cx, cy=cy)
    annulus_index = np.where(annulus_mask == 1)
    
    if null_median == True :    
        med_img = np.median(img[annulus_index])
        med_sky = np.median(sky[annulus_index])
        median_ratio = med_img / med_sky
    else :
        median_ratio = 1.
   
    img_sub = img - sky * median_ratio
    
    if verbose == True :
        print( 'Median ratio = ', median_ratio)
        plt.figure(figsize=(8,4))
        ax = plt.subplot(1,2,1)
        subimage = (img_sub*annulus_mask)[cy-radius_out:cy+radius_out+1, cx-radius_out:cx+radius_out+1]
        ax.imshow( subimage, vmin=vmin, vmax=vmax, origin='lower',cmap = plt.get_cmap('RdBu'))
        ax = plt.subplot(1,2,2)
        subimage2 = (img_sub)[cy-radius_out:cy+radius_out+1, cx-radius_out:cx+radius_out+1]
        ax.plot(subimage2[radius_out,:], 'ro', mew=0, alpha=.8)
        ax.plot(subimage2[:,radius_out], 'bo', mew=0, alpha=.8, )
        #ax.plot(img_sub[annulus_index],'ro', mew=0, alpha=.8)
        plt.ylim(vmin,vmax)
        
        print( med_img, med_sky)
    
    return img_sub
