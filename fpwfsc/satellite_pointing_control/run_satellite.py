import os, sys
import satellite_funcs as satfuncs

import fpwfsc.common.support_functions as sf
import fpwfsc.common.dm as dm
import numpy as np
import threading
from collections import deque
import matplotlib.pyplot as plt
from fpwfsc.common import fake_hardware as fhw
import ipdb

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PID import PID
from pathlib import Path
from satellite_plotter_qt import LiveSquarePlotter

def plot_square_on_image(image, center, side, theta,
                         points=None,
                         radius=None, tol=None,
                         search_center=None,
                         figsize=(6, 6), cmap='gray', title=None,
                         zoom_factor=2):
    """
    Plot a square over a 2D image using fitted parameters, with optional annulus.

    Parameters:
        image (2D ndarray): Image to display
        center (tuple): (y, x) center of the square
        side (float): Side length of the square
        theta (float): Rotation angle (radians)
        points (list of (y, x)): Optional observed points to overlay
        radius (float): Radius of annular search region (optional)
        tol (float): Tolerance of annulus (defines thickness, optional)
        search_center (tuple): (y, x) center of annular search region (optional)
        figsize (tuple): Size of the plot
        cmap (str): Colormap for image display
        title (str): Plot title
        zoom_factor (float): Size of zoom box relative to square side
    """
    c_y, c_x = center
    half = side / 2.0

    # Define corners in canonical (unrotated) coordinates
    corners = np.array([
        [-half, -half],
        [-half,  half],
        [ half,  half],
        [ half, -half],
        [-half, -half]  # to close the square
    ])

    # Apply rotation
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    rotated = corners @ rot.T
    square_yx = rotated + [c_y, c_x]

    # Plot image
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap, origin='lower')

    # Draw square
    plt.plot(square_yx[:, 1], square_yx[:, 0], 'r-', linewidth=2, alpha=0.4)

    # Mark fitted center
    plt.plot(c_x, c_y, marker='x', color='red', markersize=3, alpha=0.4, zorder=10)

    # Optional: overlay observed points
    if points:
        points = np.array(points)
        plt.plot(points[:, 1], points[:, 0], 'bx', markersize=3)

    # Optional: draw annulus centered on search_center
    if radius is not None and tol is not None:
        annulus_center = search_center if search_center is not None else (c_y, c_x)
        sc_y, sc_x = annulus_center

        outer_circle = Circle((sc_x, sc_y), radius + tol, edgecolor='cyan',
                              facecolor='none', linestyle='--', linewidth=1.5, alpha=0.3)
        inner_circle = Circle((sc_x, sc_y), radius - tol, edgecolor='cyan',
                              facecolor='none', linestyle='--', linewidth=1.5, alpha=0.3)
        plt.gca().add_patch(outer_circle)
        plt.gca().add_patch(inner_circle)

    # Zoom around the square center
    zoom_half = (side * zoom_factor) / 2
    plt.xlim(c_x - zoom_half, c_x + zoom_half)
    plt.ylim(c_y - zoom_half, c_y + zoom_half)

    if title:
        plt.title(title)
    plt.xlabel("X (col)")
    plt.ylabel("Y (row)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    camera = 'Sim'
    aosystem = 'Sim'

    script_dir = Path(__file__).parent.absolute()
    hwconfig_path = script_dir.parent / "san" / "sn_config.ini"
    hwspec_path = script_dir.parent / "san" / "sn_config.spec"

    # Print paths for debugging
    print(f"Looking for config at: {hwconfig_path}")
    print(f"Looking for spec at: {hwspec_path}")
    
    # Check if files exist
    if not hwconfig_path.exists():
        raise FileNotFoundError(f"Config file not found: {hwconfig_path}")
    if not hwspec_path.exists():
        raise FileNotFoundError(f"Config spec file not found: {hwspec_path}")

    config = 'satellite_config.ini'
    configspec = 'satellite_config.spec'

    config_file = config
    # Use the correct variable names and convert to string
    hwsettings = sf.validate_config(str(hwconfig_path), str(hwspec_path))
    settings = sf.validate_config(config, configspec)

    if camera == 'Sim' and aosystem == 'Sim':
        #----------------------------------------------------------------------
        #CAMERA, AO, CORONAGRAPH SIM SETTINGS IN CONFIG FILE
        #----------------------------------------------------------------------
        CSM      = fhw.FakeCoronagraphOpticalSystem(**hwsettings['SIMULATION']['OPTICAL_PARAMS'])
        AOSystem = fhw.FakeAODMSystem(OpticalModel=CSM, **hwsettings['SIMULATION']['AO_PARAMS'])
        Camera   = fhw.FakeDetector(opticalsystem=CSM, **hwsettings['SIMULATION']['CAMERA_PARAMS'])

    else:
        Camera = camera#hw.Camera.instance()
        AOSystem = aosystem#hw.AOSystem.instance()

    bgds = sf.setup_bgd_dict(hwsettings['CAMERA_CALIBRATION']['bgddir'])

    plotter = LiveSquarePlotter(figsize=(300,600))

    initial_shape = AOSystem.get_dm_data()
    data_nospeck_raw = Camera.take_image()
    data_nospeck = sf.equalize_image(data_nospeck_raw, **bgds)

    waffle = dm.generate_waffle(initial_shape, amplitude = 200e-9)
    AOSystem.set_dm_data(initial_shape + waffle)
    data_speck_raw = Camera.take_image()
    data_speck = sf.equalize_image(data_speck_raw, **bgds)
    # Set up the figure with toolbar for interactive navigation
    #plt.figure()
    while False:
        tiltx = float(input('What tiltx to input?'))
        tilty = float(input('What tilty to input?'))
        tt = dm.generate_tip_tilt(initial_shape.shape, tilt_x = tiltx, tilt_y=tilty, dm_rotation=0,
                                  flipy=False, flipx=False)
        AOSystem.set_dm_data(AOSystem.get_dm_data() + tt)
        data_speck_raw = Camera.take_image()
        data_speck = sf.equalize_image(data_speck_raw, **bgds)
        plt.clf()
        plt.imshow(data_speck, origin='lower'); plt.show()

        # Update the image data instead of creating a new plot
    setpointx = settings['SETPOINT']['xcen']
    setpointy = settings['SETPOINT']['ycen']
    setpoint1 = np.array([setpointy, setpointx])
    
    centerguess = np.array([setpointy, setpointx])
    searchrad   = settings['SETPOINT']['spot search radius (pix)']
    radtol      = settings['SETPOINT']['radius tolerance (pix)']
    
    
    Kp = settings['PID']['Kp']
    Ki = settings['PID']['Ki']
    Kd = settings['PID']['Kd']
    output_limit = settings['PID']['output_limits']

    PIDloop = PID(Kp=Kp, Ki=Ki, Kd = Kd, setpoint=setpoint1)
    #this should be about 5 pixels off
    tt_control = dm.generate_tip_tilt(initial_shape.shape, tilt_x = -2500e-9, tilt_y=0,
                                          dm_rotation=35, flipx=False)
    current_shape = AOSystem.get_dm_data()
    AOSystem.set_dm_data(current_shape + tt_control)
    for i in range(100):

        drift = dm.generate_tip_tilt(initial_shape.shape, tilt_x = 250e-9, tilt_y=0,
                                          dm_rotation=35, flipx=False)
        current_shape = AOSystem.get_dm_data()
        data_speck_raw = Camera.take_image()
        data_speck = sf.equalize_image(data_speck_raw, **bgds)
        points = satfuncs.find_spots_in_annulus(data_speck, centerguess, searchrad, tol=radtol,
                                             min_area=5, max_area=150)
        center, side, theta, _ = satfuncs.fit_square_and_center(points)
        plotter.update(image=data_speck,
                       setpoint=setpoint1,
                       center=center,
                       side=side,
                       theta=theta,
                       points=points,
                       radius=searchrad,
                       tol=radtol,
                       search_center=centerguess,
                       zoom_factor=3,
                       cmap='jet',
                       title='%.2f'%center[0]+', '+'%.2f'%center[1])
        if i == 0:
            import time; time.sleep(5)

        #plot_square_on_image(data_speck, center, side, theta,
        #                     points=points,
        #                     radius=searchrad, tol=radtol, search_center = centerguess,
        #                     figsize=(3,3), cmap='jet', title='%.2f'%center[0]+', '+'%.2f'%center[1])
        centerguess = center
        print("\n\n")
        print("Setpoint: ", setpoint1)
        print("Center: ", center)
        print("Error: ", setpoint1-center)
        control = 1*np.array(PIDloop(center))
        #control = (centerguess - setpoint1)*0.3
        print("control in pixel coords: ", control)
        control = control*(-100e-9)
        print("Control in t/t = ", control)
        tt_control = dm.generate_tip_tilt(initial_shape.shape, tilt_x = control[1], tilt_y=control[0],
                                          dm_rotation=35, flipx=False)
        print("\n\n")
        current_shape = AOSystem.get_dm_data()
        AOSystem.set_dm_data(current_shape + tt_control + drift)
        #print("SET DM ANGLE TO ZERO TO DEBUG THIS CRAP")
        #plt.imshow(current_shape-initial_shape);plt.colorbar();plt.show()
    plotter.execute()