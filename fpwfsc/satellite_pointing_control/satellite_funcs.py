import numpy as np
from itertools import combinations
from scipy.spatial.distance import pdist
from scipy.ndimage import median_filter, gaussian_filter
from skimage.measure import label, regionprops
import sys
import fpwfsc.common.support_functions as sf
from fpwfsc.san import sn_functions as sn_f
import ipdb

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from itertools import permutations
from scipy.optimize import minimize, Bounds

def square_fit_error_fixed_assignment(params, model_pts, data_pts):
    """
    Compute squared error given fixed correspondence between model_pts and data_pts.
    """
    residuals = model_pts[:len(data_pts)] - data_pts
    return np.sum(residuals**2)


def square_fit_error_fixed(model_pts, data_pts):
    """Squared residuals between matched model and data points."""
    return np.sum((model_pts - data_pts) ** 2)

def fit_square_and_center_3pt(points):
    """
    Robust square fit from 3 points by checking all corner matchings.
    Returns:
        center (y, x), side, theta, inferred corner
    """
    if len(points) != 3:
        raise ValueError("Function expects exactly 3 points.")

    data_pts = np.array(points)
    best_result = None
    best_error = np.inf
    best_model = None
    best_missing_index = None

    # Try all permutations of data points
    for data_perm in permutations(data_pts, 3):
        data_perm = np.array(data_perm)

        # Try all 3-point subsets of square corners (4 choose 3 = 4)
        for missing_index in range(4):
            model_indices = [i for i in range(4) if i != missing_index]

            # Initial parameter guess
            center_guess = np.mean(data_perm, axis=0)
            side_guess = np.mean([
                np.linalg.norm(data_perm[i] - data_perm[j])
                for i in range(3) for j in range(i+1, 3)
            ]) / np.sqrt(2)
            theta_guess = 0.0
            p0 = [center_guess[0], center_guess[1], max(1.0, side_guess), theta_guess]

            bounds = Bounds([0, 0, 1.0, -np.pi], [np.inf, np.inf, np.inf, np.pi])

            def cost_fn(params):
                center_y, center_x, side, theta = params
                model_all = square_model((center_y, center_x), side, theta)
                model_subset = model_all[model_indices]
                return square_fit_error_fixed(model_subset, data_perm)

            res = minimize(cost_fn, p0, method='L-BFGS-B', bounds=bounds)

            if res.success and res.fun < best_error:
                best_result = res
                best_error = res.fun
                best_model = square_model((res.x[0], res.x[1]), res.x[2], res.x[3])
                best_missing_index = missing_index

    if best_result is None:
        raise RuntimeError("3-point square fit failed.")

    inferred_corner = tuple(best_model[best_missing_index])
    return (best_result.x[0], best_result.x[1]), best_result.x[2], best_result.x[3], inferred_corner


def square_model(center, side, theta):
    """
    Generate the 4 corners of a square given center, side, and rotation.
    Returns an array of shape (4, 2) with (y, x) positions.
    """
    c_y, c_x = center
    half = side / 2.0

    # Canonical square corners centered on (0,0)
    corners = np.array([
        [-half, -half],
        [-half,  half],
        [ half,  half],
        [ half, -half]
    ])

    # Apply rotation
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    rotated = corners @ rot.T

    # Shift to center
    shifted = rotated + [c_y, c_x]
    return shifted

def square_fit_error(params, observed_pts):
    """
    Cost function: sum of squared distances between model and observed points.
    Matches observed points to closest model corners.
    """
    c_y, c_x, side, theta = params
    model_pts = square_model((c_y, c_x), side, theta)

    D = cdist(model_pts, observed_pts)
    row_ind = np.argmin(D, axis=0)  # each observed point matched to closest model point
    error = np.sum(np.min(D, axis=0) ** 2)
    return error

def fit_square_and_center(points):
    if len(points) == 4:
        return fit_square_and_center_4pt(points)
    elif len(points) == 3:
        return fit_square_and_center_3pt(points)
    elif len(points) > 4:
        best_pts = find_best_square(points)
        if best_pts is None:
            raise RuntimeError("No valid square found among candidate points.")
        return fit_square_and_center_4pt(best_pts)
    else:
        raise ValueError("Need at least 3 points to fit a square")

def fit_square_and_center_4pt(points):
    """
    Fit a square model to exactly 4 corner points using least-squares.

    Parameters:
        points: list of 4 (y, x) tuples

    Returns:
        center (y, x), side length, rotation angle (radians), None (no missing corner)
    """
    if len(points) != 4:
        raise ValueError("Exactly 4 points are required for this function.")

    pts = np.array(points)

    # Initial guesses
    center_guess = np.mean(pts, axis=0)
    side_guess = np.mean([
        np.linalg.norm(pts[i] - pts[j])
        for i in range(4) for j in range(i+1, 4)
    ]) / np.sqrt(2)
    theta_guess = 0.0
    p0 = [center_guess[0], center_guess[1], side_guess, theta_guess]

    def cost_fn(params):
        center_y, center_x, side, theta = params
        model_pts = square_model((center_y, center_x), side, theta)
        from scipy.spatial.distance import cdist
        D = cdist(model_pts, pts)
        row_ind = np.argmin(D, axis=1)
        return np.sum((model_pts - pts[row_ind]) ** 2)

    res = minimize(cost_fn, p0, method='Powell')

    if not res.success:
        raise RuntimeError("Square fit did not converge.")

    center_y, center_x, side, theta = res.x
    return (center_y, center_x), side, theta, None

def find_spots_in_annulus(
    image,
    center,
    radius,
    tol=3,
    min_area=5,
    max_area=500,
    eccentricity_thresh=0.8,
    gaussian_sigma=1.0,
    n_max=None,
    refine_centroids = False
):
    """
    Efficiently find bright, round spots near a circular radius from a central point.
    Only searches a cropped subimage to reduce processing time.

    Parameters:
        image (2D array): Input image
        center (tuple): (row, col) center point in original image
        radius (float): Expected radial distance of the spots
        tol (float): Width of the annulus to search (± tolerance)
        min_area (int): Minimum region area to be considered a spot
        max_area (int): Maximum region area
        eccentricity_thresh (float): Roundness filter (0 = circle, 1 = line)
        gaussian_sigma (float): Smoothing sigma to enhance spot detection
        n_max (int or None): If specified, returns only the n brightest spots
        refine_centroid_fn (callable or None): Optional function to refine centroids
            Should take (image, x_full, y_full) and return (y_sub, x_sub)

    Returns:
        centroids (list of tuples): Detected spot centroids in full image coordinates
    """
    row_c, col_c = center
    max_offset = int(np.ceil(radius + tol))

    # Compute subimage bounds (ensure integer slicing)
    rmin = int(max(np.floor(row_c - max_offset), 0))
    rmax = int(min(np.ceil(row_c + max_offset + 1), image.shape[0]))
    cmin = int(max(np.floor(col_c - max_offset), 0))
    cmax = int(min(np.ceil(col_c + max_offset + 1), image.shape[1]))

    # Extract subimage and preprocess
    subimage = image[rmin:rmax, cmin:cmax]
    clean = median_filter(subimage, size=3)
    smooth = gaussian_filter(clean, sigma=gaussian_sigma)

    # Create annulus mask in subimage coordinates
    rows, cols = smooth.shape
    y, x = np.indices((rows, cols))
    y_global = y + rmin
    x_global = x + cmin
    r = np.sqrt((x_global - col_c)**2 + (y_global - row_c)**2)
    mask = (r >= radius - tol) & (r <= radius + tol)

    # Threshold only the annular region
    annulus_values = smooth[mask]
    if len(annulus_values) == 0:
        return []
    threshold = np.percentile(annulus_values, 99)
    binary_mask = (smooth > threshold) & mask

    # Label and extract regions
    labels = label(binary_mask)
    props = regionprops(labels, intensity_image=smooth)

    # Filter valid blobs
    candidates = []
    for p in props:
        if min_area <= p.area <= max_area and p.eccentricity <= eccentricity_thresh:
            y_local, x_local = p.centroid
            y_full = y_local + rmin
            x_full = x_local + cmin

            if refine_centroids is True:
                refined = sn_f.get_spot_centroid(smooth, window = 20, guess_spot=(x_local, y_local))
                y_full = refined[1] + rmin  # override with refined coordinates
                x_full = refined[0] + cmin

            brightness = p.max_intensity
            candidates.append(((y_full, x_full), brightness))

    # Sort by brightness and return
    candidates.sort(key=lambda x: -x[1])  # brightest first
    centroids = [pt for pt, _ in candidates]

    if n_max is not None:
        return centroids[:n_max]
    else:
        return centroids


def is_square(pts, side_tol=0.15, diag_tol=0.15):
    """
    Check if 4 points form a square, within tolerances.
    Returns True if they do, False otherwise.
    """
    d = pdist(pts)
    d = np.sort(d)  # 6 distances: 4 sides + 2 diagonals
    sides = d[:4]
    diags = d[4:]

    # Check that all sides are nearly equal, and both diagonals match
    sides_ok = np.allclose(sides, sides[0], rtol=side_tol)
    diags_ok = np.allclose(diags, diags[0], rtol=diag_tol)
    diag_ratio = diags[0] / sides[0]

    # For a perfect square, diagonal = side * sqrt(2)
    diag_expected = np.sqrt(2) * sides[0]
    diag_match = np.isclose(diags[0], diag_expected, rtol=diag_tol)

    return sides_ok and diags_ok and diag_match

def find_square_from_points(points, side_tol=0.15, diag_tol=0.15):
    """
    Given a list of 2D points (row, col), find a group of 4 that form a square.

    Returns:
        best_square (list of 4 (row, col) tuples) or None
    """
    if len(points) < 4:
        return None

    for combo in combinations(points, 4):
        pts = np.array(combo)
        if is_square(pts, side_tol, diag_tol):
            # Optional: sort points into consistent order (e.g., clockwise from top-left)
            return sort_points_clockwise(pts)

    return None

def sort_points_clockwise(pts):
    """
    Sort 4 (row, col) points clockwise starting from top-left.
    """
    # Center the square
    c = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,0] - c[0], pts[:,1] - c[1])
    order = np.argsort(angles)
    return [tuple(pts[i]) for i in order]

def find_best_square(points, side_tol=0.102):
    """
    From a list of points (≥ 4), find the 4 that best form a square.

    Returns:
        list of 4 (y, x) tuples that form a square-like shape, or None if no valid square found.
    """
    def is_square(pts):
        pts = np.array(pts)
        dists = pdist(pts)
        dists = np.sort(dists)
        side = np.median(dists[:4])
        diag = np.median(dists[4:])
        diag_expected = np.sqrt(2) * side

        sides_ok = np.allclose(dists[:4], side, rtol=side_tol)
        diags_ok = np.allclose(dists[4:], diag_expected, rtol=side_tol)
        return sides_ok and diags_ok

    best_combo = None
    best_score = np.inf

    for combo in combinations(points, 4):
        if is_square(combo):
            center = np.mean(combo, axis=0)
            score = np.sum(np.var(combo, axis=0))  # geometric tightness
            if score < best_score:
                best_score = score
                best_combo = combo

    return list(best_combo) if best_combo else None
