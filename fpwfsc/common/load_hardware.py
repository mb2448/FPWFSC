
############################### Import Library ################################

## Math Library
import numpy as np
## System library
import sys
## Operating system library
import os
## .fits file library
import astropy.io.fits as fits
import support_functions as sf

import bench_hardware as hw

if __name__ == "__main__":
    Camera = hw.NIRC2Alias()
    AOSystem = hw.ClosedAOSystemAlias()
