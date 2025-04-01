import ipdb
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits 
from matplotlib.colors import LogNorm

def azimuthal_average(image, center=None, angular_range=None, bins=100):
    """
    Compute the azimuthal average of an image over an angular range.
    
    Parameters:
    - image: 2D numpy array representing the image (grayscale).
    - center: tuple (x, y) representing the center of the image (default is the image center).
    - angular_range: tuple (start_angle, end_angle) in degrees. The angular range to average over.
    - bins: Number of bins to divide the radial distance into.

    Returns:
    - rad_avg: The azimuthal average profile.
    - radial_bins: The corresponding radial bins.
    """
    # Image shape
    height, width = image.shape
    
    if center is None:
        center = (width // 2, height // 2)  # default to center of the image

    # Create a grid of coordinates
    Y, X = np.indices((height, width))
    Y = Y - center[1]
    X = X - center[0]
    
    # Compute the radial distance of each pixel from the center
    r = np.sqrt(X**2 + Y**2)

    # Define angular range
    if angular_range is None:
        angular_range = (0, 360)  # default range is from 0° to 360°

    start_angle, end_angle = angular_range
    start_angle_rad = np.radians(start_angle)
    end_angle_rad = np.radians(end_angle)

    # Compute the angle of each pixel
    angle = np.arctan2(Y, X)
    angle = np.degrees(angle)  # Convert angle to degrees
    angle[angle < 0] += 360  # Ensure all angles are positive
    
    # Mask pixels within the angular range
    mask = (angle >= start_angle) & (angle <= end_angle)
    
    # Radial distance bins
    radial_bins = np.linspace(0, np.max(r), bins)
    rad_avg = np.zeros(bins - 1)
    
    # Compute the azimuthal average
    for i in range(bins - 1):
        # Get pixels within the radial bin
        bin_mask = (r >= radial_bins[i]) & (r < radial_bins[i + 1]) & mask
        rad_avg[i] = np.mean(image[bin_mask]) if np.any(bin_mask) else 0
    
    return rad_avg, radial_bins[:-1]  # Return average values and radial bins


if __name__ == "__main__":

    PARENT_DIR = "output_2025-03-31_15-50-49/"
    SUFFIX = "_Halfdark_ND1_5ms.fits"
    iters = np.arange(15)
    psfs = []
    psf_profiles = []
    cx, cy = 256, 153
    cut = 64

    for i in iters:
        loaded = fits.getdata(PARENT_DIR+f"SAN_iter{i}"+SUFFIX)
        img = loaded[cy-cut:cy+cut, cx-cut:cx+cut]
        profile, bins = azimuthal_average(img)
        psfs.append(loaded)
        psf_profiles.append(profile)
    
    for i, psf in enumerate(psfs):
        plt.figure()
        plt.title(f"SAN Iteration {i}")
        plt.imshow(psf, norm=LogNorm())
        plt.xlim([cx-cut, cx+cut])
        plt.ylim([cy-cut, cy+cut])
        plt.colorbar()

    plt.figure()
    plt.title("Azavg for each iteration")
    for i, pro in enumerate(psf_profiles):
        plt.plot(bins, profile, alpha=i/15, color="k")
    plt.show()


