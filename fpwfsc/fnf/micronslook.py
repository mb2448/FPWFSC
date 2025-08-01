import numpy as np
import matplotlib.pyplot as plt
from hcipy import *
from matplotlib.colors import LogNorm

mic = np.loadtxt('microns.txt')
mic = mic.reshape((64,64))
plt.imshow(mic)
plt.show()