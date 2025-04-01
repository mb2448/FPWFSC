import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits 
from matplotlib.colors import LogNorm

if __name__ == "__main__":

    PARENT_DIR = "output_2025-03-31_13-58-45/"
    SUFFIX = "_Halfdark_ND1_5ms.fits"
    iters = np.arange(7)
    psfs = []
    cx, cy = 256, 153
    cut = 64

    for i in iters:
        loaded = fits.getdata(PARENT_DIR+f"SAN_iter{i}"+SUFFIX)
        psfs.append(loaded)
    
    for i, psf in enumerate(psfs):
        plt.figure()
        plt.title(f"SAN Iteration {i}")
        plt.imshow(psf, norm=LogNorm())
        plt.xlim([cx-cut, cx+cut])
        plt.ylim([cy-cut, cy+cut])
        plt.colorbar()
    plt.show()


