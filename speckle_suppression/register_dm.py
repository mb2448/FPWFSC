import numpy as np

def compute_angle(x, y):
    """
    Compute the angle of a point (x, y) with respect to the positive x-axis.
    
    Parameters:
    -----------
    x : float
        x-coordinate of the point
    y : float
        y-coordinate of the point
        
    Returns:
    --------
    angle : float
        Angle in degrees, measured counterclockwise from the positive x-axis.
        Range is [-180, 180] degrees.
    """
    # Use arctan2 to compute the angle
    # arctan2(y, x) returns the angle in radians in the range [-π, π]
    angle_rad = np.arctan2(y, x)
    
    # Convert to degrees
    angle_deg = np.rad2deg(angle_rad)
    
    return angle_deg

# Example usage
if __name__ == "__main__":
    # Test with the provided example
    x, y = 5, 0
    angle = compute_angle(x, y)
    print(f"The angle of point ({x}, {y}) with respect to the x-axis is {angle} degrees")
    
    # Test with a few more examples
    test_points = [
        (1, 1),    # 45 degrees
        (0, 1),    # 90 degrees
        (-1, 0),   # 180 degrees
        (0, -1),   # -90 degrees
        (1, -1)    # -45 degrees
    ]
    
    for x, y in test_points:
        angle = compute_angle(x, y)
        print(f"The angle of point ({x}, {y}) with respect to the x-axis is {angle} degrees")

