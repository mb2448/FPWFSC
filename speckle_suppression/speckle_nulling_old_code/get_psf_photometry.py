############################### Import Library ################################

## Math Library
import numpy as np
## System library
import sys
## Operating system library
import os
## .fits file library
import astropy.io.fits as fits
## Library used to plot graphics and show images
import matplotlib.pyplot as plt
# Location of the FIU library
sys.path.append('/kroot/src/kss/nirspec/nsfiu/dev/lib/')
## Function use to control NIRC2
import Nirc2_cmds as Nirc2
## Function use to get the path where data should be saved
from FIU_Commands import get_path

# Location of the Speckle Nulling library
sys.path.append('/kroot/src/kss/nirspec/nsfiu/dev/speckle_nulling/')
## Libraries to process speckle nulling data
import sn_preprocessing as pre
import sn_filehandling as flh
from dm_registration import get_satellite_centroids
from detect_speckles import create_speckle_aperture, get_speckle_photometry, \
get_total_aperture_flux

def get_optimal_aperture(rads, total_fluxes):
    """solves for the optimal aperture radius.  looks for the minimal slope
    in the total photometry in the aperture vs aperture radius in pixels.
    This should correspond to the airy ring"""
    rad_index = np.where(np.diff(total_phot) == np.min(np.diff(total_phot)))[0][0]
    optimal_rad = rads[rad_index]
    return optimal_rad -1 

if __name__ == "__main__":
    psf_image = fits.open('psf.fits')[0].data
    # Initialized config and spec file names
    soft_ini  = 'Config/SN_Software.ini'
    soft_spec = 'Config/SN_Software.spec'
    hard_ini  = 'Config/SN_Hardware.ini'
    hard_spec = 'Config/SN_Hardware.spec'
    
    # Read the config files and check if satisfied spec files requierements.
    soft_config = flh.validate_configfile(soft_ini, soft_spec)    

    print('Reading the calibration data acquired previously.')
    bgds = flh.setup_bgd_dict(soft_config)
    clean_image = pre.equalize_image(psf_image, **bgds)
    spotlocation = get_satellite_centroids(clean_image, 
                   cmt='SHIFT-Click near the center of the PSF',
                   guess_spots = [(152, 177)]
                   )[0]
    spotx, spoty = spotlocation
    print("Spot location:", '%.2f'%spotx, ', %.2f'%spoty)
    ap_rads = range(1, 10)
    mean_phot = []
    total_phot = []
    for ap_rad in ap_rads:
        aperture = create_speckle_aperture(clean_image,spotx, spoty,ap_rad)  
        mean_phot_ = get_speckle_photometry(clean_image, aperture)
        total_phot_ = get_total_aperture_flux(clean_image, aperture)
        mean_phot.append(mean_phot_)
        total_phot.append(total_phot_)
    optimal_aprad = get_optimal_aperture(ap_rads, total_phot)
    optimal_ap = create_speckle_aperture(clean_image,spotx, spoty,optimal_aprad)  
    
    #Plot the data
    fig, ax = plt.subplots(1, 3, 
                           gridspec_kw={'width_ratios':[1,1,3]},
                           figsize = (16, 7))
    ax[0].plot(ap_rads, mean_phot)
    ax[0].set_title('Mean photometry')
    ax[0].set_xlabel('Aperture rad')
    ax[0].set_ylabel('Mean counts per pixel')
    ax[1].plot(ap_rads, total_phot)
    ax[1].set_title('Total photometry')
    ax[1].set_xlabel('Aperture rad')
    ax[1].set_ylabel('Total counts in aperture')
    ax[1].axvline(optimal_aprad, color='k', linestyle='--')
    ax[2].imshow(np.log(np.abs(clean_image)), 
                 interpolation='nearest', origin='lower',
                 cmap='Greys_r', alpha = 0.2)
    ax[2].imshow(optimal_ap, alpha = 0.5,
                 interpolation='nearest', origin='lower',
                 cmap = 'jet')
    plt.title('Optimal radius:'+str(optimal_aprad)+' pixels')
    plt.tight_layout()
    plt.show()
