import numpy as np
from scipy.signal import convolve2d

def arctan(x, a=1, x0=0, y0=0):
    x = x - x0
    return np.arctan(a * x) + y0

def tanh(x, a=1, x0=0, y0=0):
    x = x - x0
    return (2 / (1 + np.exp(-2 * a * x)) - 1 + y0)

def count_neighbors(array):

    kernel = np.array([
        [1., 1., 1.],
        [1., 0., 1.],
        [1., 1., 1.],
    ])

    neighbor_count = convolve2d(array, kernel, mode="same", boundary="fill")

    return neighbor_count

def neighbor_mask(array, divisor=8):
    """wrapper to create a mask that weights a focal plane
    by the number of pixels

    Parameters
    ----------
    array : ndarray
        ndarray containing a binary control region
    divisor : float
        number to divide mask by. Defaults to 8 such that a
        pixel with 8 neighbors (completely surrounded) has
        a unity weight
    """

    neighbors = count_neighbors(array)
    return neighbors / divisor

if __name__ == "__main__":

    x = np.linspace(-1, 1, 128)
    x, y = np.meshgrid(x, x)
    r = np.hypot(x, y)
    mask = np.zeros_like(x)
    mask[r < 0.5] = 1
    neighbs = count_neighbors(mask)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(neighbs / 8)
    plt.colorbar()
    plt.show()