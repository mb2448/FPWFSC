import numpy as np
import ipdb

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
    Remove a waffle pattern from an array.
    
    This function removes the waffle component from an array by detecting
    the alternating checkerboard pattern. Works with waffle patterns generated
    by the improved generate_waffle function.
    
    Args:
        array: Input array that may contain a waffle pattern
        threshold (float, optional): Minimum amplitude to consider a waffle pattern present.
                                     If None, any detected pattern will be removed.
    
    Returns:
        numpy.ndarray: Array with waffle pattern removed
    """
    import numpy as np
    
    # Convert input to numpy array if it isn't already
    array = np.asarray(array)
    
    # Create the same checkerboard pattern used in generate_waffle
    row_indices, col_indices = np.indices(array.shape)
    checkerboard = (row_indices + col_indices) % 2
    normalized_pattern = checkerboard - 0.5  # Values of -0.5 and +0.5
    
    # Calculate the dot product to find how much of the waffle pattern exists
    dot_product = np.sum(array * normalized_pattern)
    
    # Calculate the total squared magnitude of the normalized pattern
    pattern_magnitude = np.sum(normalized_pattern * normalized_pattern)
    
    # Calculate the amplitude of the waffle component
    waffle_amplitude = dot_product / pattern_magnitude
    
    # If the amplitude is below threshold, return the original array
    if threshold is not None and abs(waffle_amplitude) < threshold:
        return array
    
    # Subtract the detected waffle component
    detected_waffle = waffle_amplitude * normalized_pattern
    result = array - detected_waffle
    
    return result

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
    Generate a waffle pattern where alternate cells have +amplitude/2 and -amplitude/2.
    
    Args:
        n_or_array: Either an integer n to create an n×n waffle,
                   or an existing array to use as a template for dimensions.
        amplitude (float, optional): Peak-to-valley amplitude of the pattern.
                                   Default is 1, which gives +0.5 and -0.5.
                                   For example, amplitude=2 gives +1 and -1.
    
    Returns:
        numpy.ndarray: An array with alternating values in a waffle pattern.
    """
    import numpy as np
    
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
    
    # Create checkerboard pattern (0s and 1s)
    checkerboard = (row_indices + col_indices) % 2
    
    # Convert to alternating -0.5 and +0.5 scaled by amplitude
    waffle = (checkerboard - 0.5) * amplitude
    
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
                    which="cos",
                    flipx=False,
                    flipy=True):
    """given an x and y pixel position,
    generates a NxN flatmap that has
    a speckle at that position"""

    #convert first to wavevector space
    kxs, kys = convert_pixels_kvecs(xs, ys,
                  centerx = centerx,
                  centery = centery,
                  angle = angle,
                  lambdaoverd = lambdaoverd)
    returnmap = make_speckle_kxy(kxs,kys,amps,phases,N=N, dm_rotation=dm_rotation, which=which,
                                 flipy=flipy, flipx=flipx)
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
