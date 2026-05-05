import numpy as np

def crop_to_square(image=None, cx=None, cy=None, size=None):
    """
    Crops an image to a square centered at specified coordinates.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image as a numpy array with shape (height, width) or (height, width, channels)
    cx : int or float
        X coordinate of the center point of the square
    cy : int or float
        Y coordinate of the center point of the square
    size : int
        Side length of the square crop

    Returns:
    --------
    tuple : (x_coords, y_coords, cropped_image)
        x_coords : numpy.ndarray (2D)
            X coordinates array of same shape as cropped image, where each element
            contains the x-coordinate of that pixel in the original image
        y_coords : numpy.ndarray (2D)
            Y coordinates array of same shape as cropped image, where each element
            contains the y-coordinate of that pixel in the original image
        cropped_image : numpy.ndarray
            The square-cropped image
    """

    if len(image.shape) == 2:
        height, width = image.shape
    elif len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        raise ValueError("Image must be 2D or 3D array")

    # Calculate half the size for offset from center
    half_size = size // 2

    # Calculate the starting coordinates for the crop
    start_x = int(cx - half_size)
    start_y = int(cy - half_size)

    # Calculate the ending coordinates
    end_x = start_x + size
    end_y = start_y + size

    # Check bounds and adjust if necessary
    if start_x < 0 or start_y < 0 or end_x > width or end_y > height:
        raise ValueError(f"Crop region ({start_x}, {start_y}) to ({end_x}, {end_y}) "
                        f"is outside image bounds (0, 0) to ({width}, {height})")

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]

    # Generate coordinate arrays that correspond to the original image coordinates
    x_range = np.arange(start_x, end_x)
    y_range = np.arange(start_y, end_y)

    # Create 2D coordinate meshgrids of the same shape as the cropped image
    x_coords, y_coords = np.meshgrid(x_range, y_range)

    return x_coords, y_coords, cropped_image

def compute_quad_cell_flux(image=None, x_center=None, y_center=None, min_radius=None, max_radius=None,
                          x_coords=None, y_coords=None):
    """
    Computes weighted flux centroid offset using quad cell method.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image array
    x_center : float
        X coordinate of center point
    y_center : float
        Y coordinate of center point
    min_radius : float
        Inner radius of annulus
    max_radius : float
        Outer radius of annulus
    x_coords : numpy.ndarray, optional
        2D array of x coordinates (same shape as image). If None, uses pixel indices.
    y_coords : numpy.ndarray, optional
        2D array of y coordinates (same shape as image). If None, uses pixel indices.

    Returns:
    --------
    tuple : (x_offset, y_offset)
        x_offset : float
            Centroid offset in x direction
        y_offset : float
            Centroid offset in y direction
    """

    # Create coordinate arrays if not provided
    if x_coords is None or y_coords is None:
        height, width = image.shape[:2]
        y_indices, x_indices = np.mgrid[0:height, 0:width]
        x_coords = x_indices if x_coords is None else x_coords
        y_coords = y_indices if y_coords is None else y_coords

    # Calculate distances from center
    dx = x_coords - x_center
    dy = y_coords - y_center
    distances = np.sqrt(dx**2 + dy**2)

    # Create annulus mask (pixels within min_radius to max_radius)
    annulus_mask = (distances >= min_radius) & (distances <= max_radius)

    # Create quadrant masks within the annulus.
    # In numpy array coordinates, increasing y-index = downward on screen.
    # dy >= 0 means larger row index = below center in display.
    # Quadrant labeling (A/B/C/D) follows array index convention, not sky.
    A_mask = annulus_mask & (dx <= 0) & (dy >= 0)  # left,  larger y-index
    B_mask = annulus_mask & (dx > 0)  & (dy >= 0)  # right, larger y-index
    C_mask = annulus_mask & (dx <= 0) & (dy < 0)   # left,  smaller y-index
    D_mask = annulus_mask & (dx > 0)  & (dy < 0)   # right, smaller y-index

    # Flux in each quadrant
    A = np.sum(image[A_mask])
    B = np.sum(image[B_mask])
    C = np.sum(image[C_mask])
    D = np.sum(image[D_mask])

    total_flux = A + B + C + D

    if total_flux == 0:
        return 0.0, 0.0

    # Centroid offsets in array-index convention:
    #   x_offset > 0 = star is at larger column index (right in display)
    #   y_offset > 0 = star is at larger row index (down in display)
    # The PID + tip-tilt calibration (gain, angle, flips) handles the
    # mapping from these offsets to the correct AO correction direction.
    y_offset = (A + B - C - D) / total_flux
    x_offset = (B + D - A - C) / total_flux

    return x_offset, y_offset


if __name__ == "__main__":
    import astropy.io.fits as pf
    test_img = pf.open('/Users/mbottom/Desktop/projects/HSF_exoplanet_imaging/fast_and_furious_wfs/code/dummy_camera_directory/image_20250916_195537_1.fits')[0].data

    xc, yc = 330, 425
    cropsize = 100
    inner_rad = 10
    outer_rad = 30

    xs, ys, cropped = crop_to_square(image=test_img, cx=xc, cy=yc, size=cropsize)

    xo, yo = compute_quad_cell_flux(image=cropped, x_center=xc, y_center=yc, min_radius=inner_rad, max_radius=outer_rad,
                          x_coords=xs, y_coords=ys)
    print(xo, yo)
