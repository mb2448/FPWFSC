import hcipy
import numpy as np
import sys
import os
import time
import astropy.io.fits as pf
from pathlib import Path
import matplotlib.pyplot as plt

class HitchhikerTimeoutError(Exception):
    """Raised when Hitchhiker times out waiting for a new image."""
    pass


class Hitchhiker:
    """
    Watches a directory for new FITS files and loads them once they appear fully written.
    Assumes all files will match the size of the first complete file.

    Args:
        imagedir (str or Path): Directory to monitor for incoming FITS files.
        poll_interval (float): Time (in seconds) between file system checks.
        timeout (float): Maximum time (in seconds) to wait for a new file.
    """

    def __init__(self, imagedir=None, poll_interval=0.5, timeout=20):
        self.watch_dir = Path(imagedir)
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.suffixes = ('.fits', '.fit')
        self.seen_files = set()
        self.reference_size = None

        if not self.watch_dir.exists():
            raise FileNotFoundError(f"Directory {self.watch_dir} does not exist.")

    def wait_for_next_image(self, timeout=None):
        """
        Waits for the next complete FITS file to appear in the watch directory.

        Args:
            timeout (float or None): Override the default timeout set at initialization.

        Returns:
            numpy.ndarray: Image data from the FITS file.

        Raises:
            HitchhikerTimeoutError: If no file is found within the timeout period.
        """
        timeout = timeout or self.timeout
        start_time = time.time()

        while True:
            new_files = sorted([
                f for f in self.watch_dir.iterdir()
                if f.suffix.lower() in self.suffixes and f not in self.seen_files
            ], key=os.path.getmtime)

            for f in new_files:
                size = f.stat().st_size
                if size == 0:
                    continue

                if self.reference_size is None:
                    if self._is_fully_written(f):
                        self.reference_size = f.stat().st_size
                        self.seen_files.add(f)

                        print(self.reference_size)
                        print(self.seen_files)
                        print(new_files)


                        return self._read_fits(f)
                    else:
                        continue

                if size == self.reference_size:
                    self.seen_files.add(f)

                    print(self.reference_size)
                    print(self.seen_files)
                    print(new_files)

                    return self._read_fits(f)
                else:
                    print(f"⚠️  File {f.name} has size {size}, expected {self.reference_size}. Trying to open anyway...")
                    try:
                        self.seen_files.add(f)
                        return self._read_fits(f)
                    except Exception as e:
                        print(f"Could not read {f.name}: {e}")
                        continue

            if timeout and (time.time() - start_time) > timeout:
                raise HitchhikerTimeoutError("No valid FITS file found within timeout.")

            time.sleep(self.poll_interval)

    def _is_fully_written(self, filepath):
        """Check if a file has stopped growing, to confirm it's finished writing."""
        try:
            size1 = filepath.stat().st_size
            time.sleep(0.2)
            size2 = filepath.stat().st_size
            return size1 == size2 and size1 > 0
        except FileNotFoundError:
            return False

    def _read_fits(self, filepath):
        """Reads and returns the primary HDU data from a FITS file."""
        with pf.open(filepath, ignore_missing_end=True, do_not_scale_image_data=True) as hdul:
            return hdul[0].data

print('ready!')

Hitch = Hitchhiker(imagedir='Hitchhiker_img')
for i in np.arange(5):
    
    img= Hitch.wait_for_next_image()
    plt.imshow(img)
    plt.show()