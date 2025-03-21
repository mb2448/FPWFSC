import numpy as np
import matplotlib.pyplot as plt


def create_annular_wedge(image, xcen, ycen, rad1, rad2, theta1, theta2):
    """
    Create an annular wedge mask and apply it to an image.
    
    Parameters:
    -----------
    image : 2D numpy array
        The input image
    xcen, ycen : int
        Central pixel coordinates
    rad1, rad2 : float
        Inner and outer radii of the annular wedge in pixels
    theta1, theta2 : float
        Start and end angles of the wedge in degrees (measured counterclockwise from the positive x-axis)
        Can use negative angles (e.g., -90 to 90 for a half-circle on the right side)
    
    Returns:
    --------
    masked_image : 2D numpy array
        Image with only the annular wedge region visible
    mask : 2D numpy array
        Binary mask of the annular wedge region
    """
    # Create coordinate grids
    y, x = np.indices(image.shape)
    
    # Calculate distance from center for each pixel
    r = np.sqrt((x - xcen)**2 + (y - ycen)**2)
    
    # Calculate angle for each pixel (in degrees)
    # arctan2 returns angles in radians in range [-π, π], so convert to degrees
    theta = np.rad2deg(np.arctan2(y - ycen, x - xcen))
    
    # Create the angle mask based on the input angles
    # No need to normalize angles - we can directly compare with the raw arctan2 output
    if theta1 <= theta2:
        # Simple case: theta1 to theta2 without crossing -180/180 boundary
        angle_mask = (theta >= theta1) & (theta <= theta2)
    else:
        # Case where the wedge crosses the -180/180 boundary
        angle_mask = (theta >= theta1) | (theta <= theta2)
    
    # Create the annular wedge mask
    mask = (r >= rad1) & (r <= rad2) & angle_mask
    
    # Apply mask to the image
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]
    
    return mask