import numpy as np
import astropy.io.fits as pf

def test_crop_visualization(test_img=None, cx=None, cy=None, size=None):
    """Test function to visualize original and cropped images with coordinate verification."""
    import matplotlib.pyplot as plt

    # Perform the crop
    x_coords, y_coords, cropped_img = crop_to_square(image=test_img, cx=cx, cy=cy, size=size)

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original image with crop region
    ax1.imshow(test_img, cmap='viridis')
    ax1.axhline(cy, color='red', linestyle='--', alpha=0.7)
    ax1.axvline(cx, color='red', linestyle='--', alpha=0.7)
    half_size = size // 2
    from matplotlib.patches import Rectangle
    rect = Rectangle((cx-half_size, cy-half_size), size, size, linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title('Original Image')

    # Cropped image with original coordinates
    ax2.imshow(cropped_img, cmap='viridis', extent=[x_coords.min(), x_coords.max(), y_coords.max(), y_coords.min()])
    ax2.axhline(cy, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(cx, color='red', linestyle='--', alpha=0.7)
    ax2.set_title('Cropped (Original Coordinates)')

    plt.tight_layout()
    plt.show()
    return x_coords, y_coords, cropped_img

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

    # Create quadrant masks within the annulus
    # Top left: dx <= 0, dy >= 0 (dy positive means higher y coordinate, which is up in display)
    top_left_mask = annulus_mask & (dx <= 0) & (dy >= 0)

    # Top right: dx > 0, dy >= 0
    top_right_mask = annulus_mask & (dx > 0) & (dy >= 0)

    # Bottom left: dx <= 0, dy < 0 (dy negative means lower y coordinate, which is down in display)
    bottom_left_mask = annulus_mask & (dx <= 0) & (dy < 0)

    # Bottom right: dx > 0, dy < 0
    bottom_right_mask = annulus_mask & (dx > 0) & (dy < 0)

    # Calculate flux in each quadrant
    A = np.sum(image[top_left_mask])      # Top left
    B = np.sum(image[top_right_mask])     # Top right
    C = np.sum(image[bottom_left_mask])   # Bottom left
    D = np.sum(image[bottom_right_mask])  # Bottom right

    # Calculate total flux
    total_flux = A + B + C + D

    # Avoid division by zero
    if total_flux == 0:
        return 0.0, 0.0

    # Calculate centroid offsets
    y_offset = (A + B - C - D) / total_flux  # Positive = shift up
    x_offset = (B + D - A - C) / total_flux  # Positive = shift right

    return x_offset, y_offset


def visualize_quad_cell(image=None, x_center=None, y_center=None, min_radius=None, max_radius=None, x_coords=None, y_coords=None):
    """Simple visualization of quad cell mask overlay."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    # Create coordinate arrays if not provided
    if x_coords is None or y_coords is None:
        height, width = image.shape[:2]
        y_indices, x_indices = np.mgrid[0:height, 0:width]
        x_coords = x_indices if x_coords is None else x_coords
        y_coords = y_indices if y_coords is None else y_coords

    # Show image with mask outline
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='viridis', extent=[x_coords.min(), x_coords.max(), y_coords.max(), y_coords.min()])

    # Add circles and crosshair
    circle_inner = Circle((x_center, y_center), min_radius, fill=False, color='red', linewidth=2)
    circle_outer = Circle((x_center, y_center), max_radius, fill=False, color='red', linewidth=2)
    plt.gca().add_patch(circle_inner)
    plt.gca().add_patch(circle_outer)
    plt.axhline(y_center, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x_center, color='red', linestyle='--', alpha=0.7)
    plt.plot(x_center, y_center, 'r+', markersize=10, markeredgewidth=2)

    plt.title('Quad Cell Mask Overlay')
    plt.show()


if __name__ == "__main__":
    test_img = pf.open('/Users/mbottom/Desktop/projects/HSF_exoplanet_imaging/fast_and_furious_wfs/code/dummy_camera_directory/image_20250916_195537_1.fits')[0].data

    xc, yc = 330, 425
    cropsize = 100
    inner_rad = 10
    outer_rad = 30

    xs, ys, cropped = crop_to_square(image=test_img, cx=xc, cy=yc, size=cropsize)

    #test_crop_visualization(test_img=test_img, cx=xc, cy=yc, size=cropsize)

    #visualize_quad_cell(image=test_img, x_center=xc, y_center=yc, min_radius=inner_rad, max_radius=outer_rad)
    visualize_quad_cell(image=cropped, x_center=xc, y_center=yc, min_radius=inner_rad, max_radius=outer_rad, x_coords=xs, y_coords=ys)
    xo, yo = compute_quad_cell_flux(image=cropped, x_center=xc, y_center=yc, min_radius=inner_rad, max_radius=outer_rad,
                          x_coords=xs, y_coords=ys)
    print(xo, yo)
