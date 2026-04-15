"""
Interactive NIRC2 image grabber.

Run with:  python -i nirc2_image_grabber.py

Then call save_image('/path/to/dir') whenever you want a new FITS
dropped into that directory for hitchhiker mode to pick up.
"""

import os
import time

import astropy.io.fits as pf

from fpwfsc.qacits import qacits_hardware

camera = qacits_hardware.NIRC2Alias()


def save_image(directory, name=None):
    """Take one image and write it to `directory` as a FITS file."""
    os.makedirs(directory, exist_ok=True)
    data = camera.take_image()
    if name is None:
        name = f"nirc2_{time.strftime('%Y%m%d_%H%M%S')}.fits"
    path = os.path.join(directory, name)
    pf.writeto(path, data, overwrite=True)
    print(f"wrote {path}")
    return path
