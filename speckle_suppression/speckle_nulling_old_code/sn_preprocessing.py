############################### Import Library ################################

## Math Library
import numpy as np
## Median filter function
from scipy.ndimage import median_filter as medfilt
## Library used to plot graphics and show images
import matplotlib.pyplot as plt
## Curve fit function
from scipy.optimize import curve_fit
import pdb
##Are those libraries used?

## Operating system library
import os

import astropy.io.fits as pf
from scipy.special import gamma
import scipy.ndimage as sciim
import sn_math as snm

###############################################################################

# =============================================================================
def removebadpix(data, mask, kern = 5):
    ''' -----------------------------------------------------------------------
    Removes bad pixels by replacing them with a kern x kern  kernel median 
    filter of the image where they exist.
    ----------------------------------------------------------------------- '''
    # Create a copy of the provided data
    tmp = data.copy()
    # Compute the medfilt image associated to the provided data
    medianed_image = medfilt(tmp, size=(kern, kern), mode='wrap')
    # Replaces the bad pixel by the computed value 
    tmp[np.where(mask>0)] = medianed_image[np.where(mask>0)]
    # Return the cleaned data
    return tmp

# =============================================================================
def locate_badpix(data, sigmaclip = 5):
    ''' -----------------------------------------------------------------------
    Locates bad pixels by fitting a gaussian distribution to the image
    intensity and then cutting outliers at the level of 'sigmaclip'
    ----------------------------------------------------------------------- '''
    # Create a vector of values borned by the min and max of the provided data
    xvals = np.arange(data.min(), data.max())
    # Short the value of the provided data to create an histogram
    yvals = np.histogram(data.ravel(), bins=xvals, density=True)[0]
    # Find the position associated to the faintest and brightest points.
    m1 = np.abs(np.cumsum(yvals)-0.0005).argmin()
    m2 = np.abs(np.cumsum(yvals)-0.9995).argmin()
    # Compute the mean of the selected points
    midx = 0.5*(xvals[m1]+xvals[m2])
    # Reduce the list of points by removing the faintest and brightest points.
    tmpx = xvals[m1:m2]
    tmpy = yvals[m1:m2]
    # Fit a Gaussian on the selected points
    popt, pcov = curve_fit(gaussfunc, tmpx, tmpy, p0 = (midx,25))
    # Extract the mean and standard deviation
    mean   = popt[0]
    stddev = popt[1]
    # Compute the brighntness limits of the point to keep
    cliphigh  = mean + sigmaclip*np.abs(stddev)
    cliplow   = mean - sigmaclip*np.abs(stddev)
    # Generate the bad pixel map 
    bpmask = np.round(data > cliphigh) + np.round(data < cliplow)
    # Plot the histogram
    plt.plot(xvals[:-1], yvals)
    # Overplot gaussian fit on the data
    plt.plot(xvals[:-1], gaussfunc(xvals[:-1], *popt))
    # Add labels
    plt.xlabel('Number of pixels')
    plt.ylabel('Pixel intensities')
    # Prapare the title of the figure
    title  = 'Fit to bad pixels' 
    title += '\nOutside shaded area pixels are considered bad'
    title += '\n Close this window to continue'
    # Add the title
    plt.title(title)
    # Highlight pixels considered as good pixels
    plt.axvspan(cliplow, cliphigh, alpha=0.2, color='grey')
    # Show the figure
    plt.show()
    # Return the bad pixel mask as a numpy array.
    return np.array(bpmask, dtype=np.float32)

# =============================================================================
def get_spot_locations(image, comment = None, eq = True):
    ''' -----------------------------------------------------------------------
    Get the position of a click in an image.
    Note this is in imshow coordinates where the bottom left pixel is (-.5,-.5)
    ----------------------------------------------------------------------- '''
    # =========================================================================
    class EventHandler:
        ''' -------------------------------------------------------------------
        This local class is used to roughtly determine the position of object 
        of interest displayed previously. The user must right click on the 
        objects of interest by following the instruction provided (see title of
        the figure. Then the user has to close the figure.
        ------------------------------------------------------------------- '''        
        # =====================================================================
        def __init__(self, spotlist):
            ''' ---------------------------------------------------------------
            --------------------------------------------------------------- '''            
            fig.canvas.mpl_connect('button_press_event'  , self.onpress       )
            fig.canvas.mpl_connect('key_press_event'     , self.on_key_press  )
            fig.canvas.mpl_connect('key_release_event'   , self.on_key_release)   
            # Initialize the shift keyboard key status       
            self.shift_is_held = False

        # =====================================================================
        def on_key_press(self, event):
            ''' ---------------------------------------------------------------
            This function detect when the keybord key 'shift' is held.
            --------------------------------------------------------------- '''
            
            if event.key == 'shift':
                self.shift_is_held = True

        # =====================================================================
        def on_key_release(self, event):
            ''' ---------------------------------------------------------------
            This function detect when the keybord key 'shift' is not held.
            --------------------------------------------------------------- '''
            if event.key == 'shift':
                self.shift_is_held = False

        # =====================================================================
        def onpress(self, event):
            ''' ---------------------------------------------------------------
            Determine and display the position where the user click on the fig.
            --------------------------------------------------------------- '''
            # Check if the user clicked in the limits of the figure displayed.
            if event.inaxes != ax:
                # If not the function does not return anything. 
                return
            # If yes, check if the shift keyboard key was held simultaneously.
            if self.shift_is_held:
                # Get the position in pixel
                xi, yi = (int(round(n)) for n in (event.xdata, event.ydata))
                # Get the value associated to the position
                value  = im.get_array()[xi,yi]
                # ?
                color  = im.cmap(im.norm(value))
                # Add the position to the spotlist
                spotlist.append((xi, yi))
                # Get the iteration number
                ite    = len(spotlist)
                # Print the iteration number and the position in pixel
                print('Postion #%02d : x = %07.3f -- y = %07.3f ' %(ite,xi,yi))

    # =========================================================================
    # Function starts here
    # =========================================================================
    
    # Case #1: User want ot display the image in log scale
    if eq:
        # Prepare image to be displayed
        image = np.log(np.abs(np.copy(image)))
        # Display the image
        im = plt.imshow(image, interpolation = 'nearest', origin = 'lower')

    # Case #2: User want ot display the image in linera scale
    else:
        im = plt.imshow(image, interpolation='nearest', origin='lower')

    # Define the standard title
    std_title = ('SHIFT-Click on spot(s). Close when finished')

    # Try to add the title to the figure. 
    # Add the standard title if failed or if not defined.
    try:
        if comment is None:
            plt.title(std_title)
        else:
            plt.title(comment)
    except:
        plt.title(std_title)

    # Defines the fig and axis variabled for the EvenHandler class
    fig = plt.gcf()
    ax  = plt.gca()
    
    # Initialize the list of spot position for the EvenHandler class
    spotlist = []
    # Ask the user to roughly locate the speckles
    handler = EventHandler(spotlist) 
    # Display the figure
    plt.show()
    # Return the list of speckle found by the user.
    return spotlist

# =============================================================================
def subimage(im, center, window = 20):
    ''' -----------------------------------------------------------------------
    Returns a subimage of size 'window' x window centered on the (x,y) pixel 
    position passed as 'center'
    ----------------------------------------------------------------------- '''
    # Extract x and y position
    y0 , x0 = center
    y0 = int(round(y0))
    x0 = int(round(x0))
    window = int(round(window))
    # Compute the limits of the sub images
    xmin = x0 - round(window/2)
    xmax = x0 + round(window/2)
    ymin = y0 - round(window/2)
    ymax = y0 + round(window/2)
    # Return the sub image
    return im[xmin:xmax, ymin:ymax]










def linearize_and_align(image):
    """Not used: apply Stan Metchev's distortion correction to PHARO"""
    coeff=np.array([[0.9994,0.56e-7,-2.70e-11],  
                    [1.0033,-6.60e-7,-9.80e-12], 
                    [1.0010, -3.42e-7, -1.72e-11], 
                    [1.0011,-5.15e-7,-1.29e-11]])

def poissfunc(x, mu):
    """Poisson distribution"""
    return np.exp(-1*mu)*mu**(x)/gamma(x+1)

def gaussfunc(x,  mu, sig):
    """1-d Gaussian x, mu, sigma"""
    return (1.0/(sig*np.sqrt(2*np.pi))*
            np.exp(-(x-mu)**2/(2*sig**2)))


def combine_quadrants(image):
    #"""combines the four camera quadrants into a unified image"""
    data = image[0].data
    ##in this program WYSIWYG with CAMERA;s monitor
    ##returnimage = returnimage[:,::-1]
    #return returnimage
    return data


def histeq(im,nbr_bins=256):
    """histogram equalize an image"""
    #get image histogram
    im = np.abs(im)
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)

    return im2.reshape(im.shape)

def equalize_image(data, bkgd=None, masterflat=None, badpix=None):
     """removes bad pixels from data, remove bad pixels from background,
       subtracts the two, and divides by the master flat field"""
    #return removebadpix(data-bkgd, badpix)/masterflat
     return removebadpix(data-bkgd, badpix)/masterflat



def buildcube(filelist):
    """Takes a file list (of fits files) and constructs a single
       data cube out of them"""
    xyshape = np.shape(pf.open(filelist[0])[0].data)
    precube = np.zeros( (len(filelist), xyshape[0], xyshape[1]))
    for idx, fitsfile in enumerate(filelist):
        with pf.open(fitsfile) as hdulist:
            precube[idx, :,:]=hdulist[0].data
    return precube

def quickalign(datacube, window=20):
    """quickly aligns a datacube to the first image 
       around a user-clicked subwindow"""
    #note this does this in the opposite order
    #from the calculate_offsets since this one 
    #assumes you don't care too much
    aligned_datacube = np.zeros(datacube.shape)
    firstimage = datacube[0,:,:]
    aligned_datacube[0,:,:]=firstimage

    spotlist = get_spot_locations(firstimage)
    for i in range(1, datacube.shape[0]):
        offsets = 'None' 
        for spot in spotlist:
            froi    = subimage(firstimage, spot, window=window)
            froi    = medfilt(froi, size = (3, 3), mode='wrap')
            roi    = subimage(datacube[i,:,:], spot, window=window)
            roi    = medfilt(roi, size = (3, 3), mode='wrap')
            offset = np.array(imreg.chi2_shift(froi,roi))
            if offsets=='None':
                offsets = offset
            else:
                offsets = np.vstack((offsets, offset))
        if offsets.ndim >1:
            moffset = np.mean(offsets, axis = 1)
        else:
            moffset = offset
        shifted = sciim.interpolation.shift(datacube[i,:,:],
                             [-1*moffset[1],-1*moffset[0]], order = 1)
        aligned_datacube[i,:,:]=shifted
    return aligned_datacube

def crop(image, xmin, xmax, ymin, ymax):
    return image[xmin:xmax, ymin:ymax]



def subimagecube(cubedata, cx, cy, window = 20):
    """returns a subcube of size 'window' about a certain (x,y) pixel, 
       passed as 'center'"""
    hw = round(window/2)
    xmin = cx - hw
    xmax = cx + hw
    ymin = cy - hw
    ymax = cy + hw
    return cubedata[:, xmin:xmax, ymin:ymax]

def unsubimagecube(cube, newcubedata, cx, cy, window=20):
    """reinserts a subcube into the original cube"""
    hw = round(window/2)
    copycube = cube.copy()
    copycube[:, cx-hw:cx+hw, cy-hw:cy+hw]=newcubedata
    return copycube

def unsubimage(oldim, newimdata, cx, cy, window= 20):
    """reinserts a image into the original cube"""
    hw = round(window/2)
    oldimcopy = oldim.copy()
    oldimcopy[cx-hw:cx+hw, cy-hw:cy+hw]=newimdata
    return oldimcopy


def threshold(imagedata):
    """crude threshold, don't do it"""
    maxl = np.max(imagedata)
    return imagedata*(imagedata>.8*maxl)

def quickcentroid(image, window = 20):
    """quick gaussian centroid of a spot"""
    xy = get_spot_locations(image, comment='Click Quick Centroid')[0]
    subim = subimage(image, xy, window=window)
    popt = plm.image_centroid_gaussian1(subim)
    xcenp = popt[1]
    ycenp = popt[2]
    xcen = xy[0]-round(window/2)+xcenp
    ycen = xy[1]-round(window/2)+ycenp
    return (xcen, ycen)

def quick2dgaussfit(image, window = 20, xy = None):
    """return the parameters of a 2d gaussian fit to an image.
       the gaussian model is in pl_math called image_centroid_gaussian1"""
    if xy is None:
        xy = get_spot_locations(image, comment='Click Quick Gaussian')[0]
       
    subim = subimage(image, xy, window=window)
    popt = plm.image_centroid_gaussian1(subim)
    xcenp = popt[1]
    ycenp = popt[2]
    xcen = xy[0]-round(window/2)+xcenp
    ycen = xy[1]-round(window/2)+ycenp
    #popt[1]=xcen
    #popt[2]=ycen
    return popt

def quick2dairyfit(image, window=20):
    """return the parameters of a 2d airy fit to an image.
       the airy model is in pl_math called image_centroid_airy1"""
    xy = get_spot_locations(image, comment = 'Click Quick Airy')[0]
    subim = subimage(image, xy, window=window)
    popt = plm.image_centroid_airy(subim)
    xcenp = popt[1]
    ycenp = popt[2]
    xcen = xy[0]-round(window/2)+xcenp
    ycen = xy[1]-round(window/2)+ycenp
    popt[1]=xcen
    popt[2]=ycen
    return popt



def get_spot_locations2(refimage, comment=None, eq=True):
    """get the position of a click in an image; note this is in 
       imshow coordinates where the bottom left pixel is (-.5, -.5)"""
    fig44 = plt.figure()
    plt.show()
    plt.ioff()
    class EventHandler:
        def __init__(self, spotlist):
            fig44.canvas.mpl_connect('button_press_event', self.onpress)
            fig44.canvas.mpl_connect('key_press_event', self.on_key_press)
            fig44.canvas.mpl_connect('key_release_event', self.on_key_release)          
            self.shift_is_held = False
        def on_key_press(self, event):
           if event.key == 'shift':
               self.shift_is_held = True

        def on_key_release(self, event):
           if event.key == 'shift':
               self.shift_is_held = False

        def onpress(self, event):
            if event.inaxes!=ax:
                return
            if self.shift_is_held:
                xi, yi = (int(round(n)) for n in (event.xdata, event.ydata))
                value = im.get_array()[xi,yi]
                color = im.cmap(im.norm(value))
                print(xi,yi)
                spotlist.append((xi, yi))
                print(spotlist)
    if eq:
        im = plt.imshow(np.log(np.abs(refimage)), interpolation='nearest', origin='lower')
    else:
        im = plt.imshow(refimage, interpolation='nearest', origin='lower')
    if comment is None:
        comment = ('SHIFT-Click on spot(s) to align to. Close when finished')
    try:
        plt.title(comment)
    except:
        pass
    fig44 = plt.gcf()
    ax = plt.gca()
    
    spotlist = []
    #Pick initial spots
    handler=EventHandler(spotlist) 
    plt.show()
    return spotlist
