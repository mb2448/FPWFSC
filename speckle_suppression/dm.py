import numpy as np

def rotateXY(xvals, yvals, thetadeg = 0):
    theta = np.pi/180.0*thetadeg
    return (np.cos(theta)*xvals - np.sin(theta)*yvals, 
            np.sin(theta)*xvals + np.cos(theta)*yvals)

def intensitymodel( amp, k_rad, a=0, b=0, c=0):
    """Radial dependence of spot calibration\n
    intensity = amp**2*(a*k_rad**2 + b*k_rad + c)"""
    return  amp**2*(a*k_rad**2 + b*k_rad + c)

def amplitudemodel(counts, k_rad, a=0, b=0, c=0):
    """Radial dependence of spot calibration\n
    amplitude = sqrt(counts/(a*k_rad**2 + b*k_rad + c))""" 
    #fudge = 0.5
    fudge = 1
    retval = fudge*np.sqrt((counts/(a*k_rad**2 + b*k_rad + c)))
    if np.isnan(retval):
        return 0
    else:
        return retval

def remove_waffle(array, threshold=None):
    """
    Remove waffle pattern from an array by detecting and subtracting it.
    
    This function detects if a waffle pattern exists in the input array
    and removes it by subtracting the estimated waffle component.
    
    Args:
        array: Input array that may contain a waffle pattern.
        threshold (float, optional): Minimum amplitude to consider a waffle pattern present.
                                    If None, any detected pattern will be removed.
    
    Returns:
        numpy.ndarray: Array with waffle pattern removed.
    """
    # Convert input to numpy array if it isn't already
    array = np.asarray(array)
    
    # Create row and column indices
    row_indices, col_indices = np.indices(array.shape)
    
    # Generate waffle pattern mask (1s and 0s)
    waffle_mask = (row_indices + col_indices) % 2
    
    # Split array into even and odd cells (according to waffle pattern)
    even_cells = array[waffle_mask == 1]
    odd_cells = array[waffle_mask == 0]
    
    # Calculate the average value for even and odd cells
    even_mean = np.mean(even_cells)
    odd_mean = np.mean(odd_cells)
    
    # Calculate the amplitude of the waffle pattern
    waffle_amplitude = even_mean - odd_mean
    
    # If the amplitude is below threshold, return the original array
    if threshold is not None and abs(waffle_amplitude) < threshold:
        return array
    
    # Create a waffle pattern with the detected amplitude
    # For even cells: add waffle_amplitude/2, for odd cells: subtract waffle_amplitude/2
    correction = (waffle_mask - 0.5) * waffle_amplitude
    
    # Remove the waffle pattern by subtracting the correction
    corrected_array = array - correction
    
    return corrected_array

def generate_tip_tilt(shape, tilt_x=0, tilt_y=0, flipx=False, flipy=False, dm_rotation=0):
    """
    Generates a tip/tilt (linear slope) waveform with specified x and y components.
    
    Parameters
    ----------
    shape : tuple
        Shape of the output array (ny, nx)
    tilt_x : float, optional
        Amplitude of the tilt in the x-direction (peak-to-valley)
    tilt_y : float, optional
        Amplitude of the tilt in the y-direction (peak-to-valley)
    flipx : bool, optional
        Whether to flip the x-component of the tip/tilt
    flipy : bool, optional
        Whether to flip the y-component of the tip/tilt
    dm_rotation : float, optional
        Rotation of the DM about the propagation axis in degrees
        
    Returns
    -------
    numpy.ndarray
        Array containing the tip/tilt waveform
    """
    # Get array dimensions
    ny, nx = shape
    
    # Create normalized coordinate grids from -0.5 to 0.5
    x, y = np.meshgrid(
        np.linspace(-0.5, 0.5, nx),
        np.linspace(-0.5, 0.5, ny)
    )
    
    # Apply DM rotation to the coordinates
    x_rot, y_rot = rotateXY(x, y, thetadeg=dm_rotation)
    
    # Apply flips if requested
    fx = -1 if flipx else 1
    fy = -1 if flipy else 1
    
    # Create the tip/tilt surface
    # Separate x and y components allow for independent control
    tip_tilt = fx * tilt_x * x_rot + fy * tilt_y * y_rot
    
    return tip_tilt

def generate_waffle(n_or_array, amplitude=1):
    """
    Generate a waffle pattern where every other cell is 1 or 0.
    
    Args:
        n_or_array: Either an integer n to create an n×n waffle,
                   or an existing array to use as a template for dimensions.
        amplitude (float, optional): Value to multiply the pattern by.
                                   Default is 1, which gives 1s and 0s.
                                   For example, amplitude=2 gives 2s and 0s.
    
    Returns:
        numpy.ndarray: An array with alternating values in a waffle pattern.
    """
    # Determine the dimensions
    if isinstance(n_or_array, (int, float)):
        # If n_or_array is a number, create an n×n array
        n = int(n_or_array)
        # Create a meshgrid of row and column indices
        row_indices, col_indices = np.indices((n, n))
    else:
        # If n_or_array is an array, use its dimensions
        template_array = np.asarray(n_or_array)
        shape = template_array.shape
        # Create a meshgrid of row and column indices for the given shape
        row_indices, col_indices = np.indices(shape)
    
    # A cell is 1 if the sum of its row and column indices is even
    # and 0 if the sum is odd
    waffle = (row_indices + col_indices) % 2
    
    # Apply the amplitude
    if amplitude != 1:
        waffle = waffle * amplitude
    
    return waffle

def make_speckle_kxy(kx, ky, amp, phase, N=21, flipy = True, flipx = False, dm_rotation=0, which="cos"):
    """given an kx and ky wavevector, 
    generates a NxN flatmap that has 
    a speckle at that position
    
    Parameters
    ----------
    kx : float or ndarray
        x-component of the wavevector. If ndarray, must be same shape as ky 
        and output appends a dimension of size kx.shape[0]
    ky : float or ndarray
        y-component of the wavevector. If ndarray, must be same shape as kx
    amp: float 
        amplitude in physical units of meters
    phase: float
        phase in radians
    dm_rotation : float
        rotation of the DM about the propagation axis, degrees
    which : str
        "sin" or "cos", determines whether the speckle is a sine or a cosine
    """

    if which == "cos":
        sinusoid = np.cos
    elif which == "sin":
        sinusoid = np.sin
    else:
        raise ValueError(f"kwarg 'which'={which} invalid, use 'sin' or 'cos'")

    
    dmx, dmy   = np.meshgrid( 
                    np.linspace(-0.5, 0.5, N),
                    np.linspace(-0.5, 0.5, N))
    
    xm=dmx*kx*2.0*np.pi
    ym=dmy*ky*2.0*np.pi

    xm, ym = rotateXY(xm, ym, thetadeg=dm_rotation)
    
    fx = -1 if flipx else 1
    fy = -1 if flipy else 1
    
    ret = amp*sinusoid(fx*xm + fy*ym +  phase)
    return ret

def make_speckle_xy(xs, ys, amps, phases, 
                    centerx=None, centery=None, 
                    angle = None,
                    lambdaoverd= None,
                    N=22,
                    dm_rotation=0,
                    which="cos"):
    """given an x and y pixel position, 
    generates a NxN flatmap that has 
    a speckle at that position"""
    #convert first to wavevector space
    kxs, kys = convert_pixels_kvecs(xs, ys, 
                  centerx = centerx,
                  centery = centery,
                  angle = angle,
                  lambdaoverd = lambdaoverd)
    returnmap = make_speckle_kxy(kxs,kys,amps,phases,N=N, dm_rotation=dm_rotation, which=which)
    return returnmap

def convert_pixels_kvecs(pixelsx, pixelsy, 
                    centerx=None, centery=None, 
                    angle = None,
                    lambdaoverd= None):
    """converts pixel space to wavevector space"""
    offsetx = pixelsx - centerx
    offsety = pixelsy - centery

    rxs, rys = rotateXY(offsetx, offsety, 
                            thetadeg = -1.0*angle)
    kxs, kys = rxs/lambdaoverd, rys/lambdaoverd
    return kxs, kys
                     
def convert_kvecs_pixels(kx, ky, 
                    centerx=None, centery=None, 
                    angle = None,
                    lambdaoverd= None):
    """converts wavevector space to pixel space"""
    rxs, rxy = kx*lambdaoverd, ky*lambdaoverd
    offsetx, offsety = rotateXY(rxs, rxy, 
                                    thetadeg = angle)
    pixelsx = offsetx + centerx
    pixelsy = offsety + centery
    return pixelsx, pixelsy

if __name__ == "__main__":
    print("todo")
    
    
    
