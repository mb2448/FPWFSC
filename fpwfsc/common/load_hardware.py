
############################### Import Library ################################

## Math Library
import numpy as np
## System library
import sys
## Operating system library
import os
## .fits file library
import astropy.io.fits as fits
from fpwfsc.common import support_functions as sf

from fpwfsc.common import bench_hardware as hw

if __name__ == "__main__":
    #Camera = hw.NIRC2Alias()
    AOSystem = hw.ClosedAOSystemAlias()

