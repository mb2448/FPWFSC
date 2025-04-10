import sys
sys.path.insert(0, '/Users/mbottom/Desktop/projects/HSF_exoplanet_imaging/fast_and_furious_wfs/code/FPWFSC/speckle_suppression')
import numpy as np
import astropy.io.fits as pf
import ipdb
import os
import configobj as co
import sn_functions as sn_f
import matplotlib.pyplot as plt
from matplotlib import ticker
#import sn_filehandling as flh
import scipy.ndimage.filters as filt
import glob

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



def build_cleancube(directory):
    ref   = glob.glob(os.path.join(directory, 'ref_img*.fits'))
    files = glob.glob(os.path.join(directory, 'SAN_iter*.fits'))
    sorted_files = [file for file in files if '_dm_shape.fits' not in file]
    sorted_files = sorted(sorted_files, key=lambda x: int(x.split('_iter')[1].split('_')[0]))
    sorted_files.insert(0, ref[0])
    output = []
    for file in sorted_files:
        try:
            data = pf.open(file)[0].data
            output.append(data)
        except FileNotFoundError as e:
            print(e)

    return np.array(output)

def build_calcube(directory):
    output = []
    files = ['controlregion.fits', 'medbackground.fits', 'masterflat.fits', 'badpix.fits']
    for f in files:
        datafile = os.path.join(directory, f)
        data = pf.open(datafile)[0].data
        output.append(data)
    return np.array(output)

def display(cleancube, controlregion, max_frame=None, display=None):
    ccdisp = cleancube.copy()
    if max_frame is None:
        max_frame = ccdisp.shape[0]
    region = controlregion
    regionboundary = filt.laplace(region)
    for i in range(ccdisp.shape[0]):
        jq = ccdisp[i,:,:]
        jq[np.where(regionboundary)]=np.nan
        ccdisp[i,:,:]=jq
    if display:
        flh.ds9(ccdisp[0:max_frame,:,:])
    return ccdisp[0:max_frame,:,:]

def ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    if exp == 0 or exp == 1:
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))

def neighbors(image, x, y):
    X = image.shape[0]-1
    Y = image.shape[1]-1
    myneighbors = lambda x, y : [(x2, y2) for x2 in range(x-1, x+2)
                           for y2 in range(y-1, y+2)
                           if (-1 < x <= X and
                               -1 < y <= Y and
                               (x != x2 or y != y2) and
                               (0 <= x2 <= X) and
                               (0 <= y2 <= Y))]
    return myneighbors(x,y)

def interpolatenans(X):
    nans = np.where(~np.isfinite(X))
    for nan in zip(*nans):
        nanx, nany = nan[0], nan[1]
        neighbs = neighbors(X, nanx, nany)
        X[nanx, nany] = np.ma.masked_invalid(
                            X[map(list, zip(*neighbs))]).mean()
    return X

def savefigandcode():
    plt.savefig()

if __name__ == "__main__":
    #directory = '/Users/mbottom/Desktop/projects/HSF_exoplanet_imaging/keck_speckle_nulling/Results/nulling_onskyrun_march2021/20210324-024716/'
    directories = ["/Users/mbottom/Downloads/output_2025-04-02_11-43-36"]
    for directory in directories:
        print(directory)
        #sys.exit(1)
        #cleanfile = glob.glob(os.path.join(directory, '*clean*.fits'))[0]
        cleancube = build_cleancube(directory)
        controlregion = pf.open(os.path.join(directory, 'controlregion.fits'))[0].data
        configfile = glob.glob(os.path.join(directory, 'sn_config.ini'))[0]
        config = co.ConfigObj(configfile)
        cx = int(float(config['DM_REGISTRATION']['MEASURED_PARAMS']['centerx']))
        cy = int(float(config['DM_REGISTRATION']['MEASURED_PARAMS']['centery']))
        l_overd = float(config['DM_REGISTRATION']['MEASURED_PARAMS']['lambdaoverd'])

        doublesided = True
        if doublesided == True:
            controlregion = controlregion + flip_array_about_point(controlregion, cx, cy)
        #Scale data by coronagraph attenuation using an off-coronagraph PSF
        scalefactor = 1.0#/7367*(.00170/.02095)*1
        cleancube = cleancube*scalefactor

        valid_data = np.mean(cleancube, axis=(1,2))>0
        max_curves=np.sum(valid_data)
        colors=plt.cm.rainbow(np.linspace(0,1,max_curves))
        fig, ax = plt.subplots(figsize = (20, 10))
        for i in range(max_curves):
            print( i)
            frame = cleancube[i,:,:]
            #frame = interpolatenans(frame)
            pixrad, clevel = sn_f.contrastcurve_simple(frame,
                                            cx=cx,
                                            cy=cy,
                                            region = controlregion,
                                            robust = True,
                                            maxrad = 10*l_overd)
            if i ==0:
                plt.plot(pixrad/l_overd, clevel, alpha = 1,
                            label = 'Initial', color='Black')
            else:
                plt.plot(pixrad/l_overd, clevel, alpha = 0.5,
                            label = 'Iteration '+str(i), color=colors[i])
        plt.axhline(y=sn_f.robust_sigma(frame[0:50, 0:50].ravel()), label='Background limit',
                    linestyle = '--',color='black')
        plt.legend(prop={'size':15})
        #plt.xlim((3, 25))
        #plt.ylim((.000025, .0005))
        #plt.yticks([.00005, .0001, .0005],
        #           ['5 10$^{-5}$', '1 10$^{-4}', '5 10$^{-4}$'])
        ax.set_yscale('log')
        plt.xlabel('Radial diffraction beamwidths away ($\lambda/D$)')
        plt.ylabel('Raw 1-sigma contrast')
        ax.grid(b=True, which = 'major')
        ax.grid(b=True, which='minor')
        subs = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]  # ticks to show per decade
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs=subs)) #set the ticks position
        ax.yaxis.set_major_formatter(ticker.NullFormatter())   # remove the major ticks
        ax.yaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))  #add the custom ticks
        plt.tight_layout()
        plt.savefig(os.path.join(directory,'fig_contrastinregion.png'), dpi = 300)

        thisfilename = sys.argv[-1]
        with open(thisfilename) as f:
            a = f.readlines()
        with open(os.path.join(directory, 'fig_code.txt'), 'w') as f:
            for l in a:
                f.write(l)
        out = display(cleancube, controlregion, max_frame=max_curves, display=False)
        hdu =pf.PrimaryHDU(out)
        hdu.writeto(os.path.join(directory,'displaycube.fits'), overwrite=True)
