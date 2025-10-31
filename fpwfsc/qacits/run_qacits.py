import os, sys

import numpy as np
import threading
from collections import deque
import queue  # For thread-safe setpoint updates

from fpwfsc.common import fake_hardware as fhw
import fpwfsc.common.support_functions as sf
import fpwfsc.common.dm as dm

from pathlib import Path
from fpwfsc.qacits.qacits_plotter_qt import QacitsPlotter
import fpwfsc.qacits.qacits_funcs as qf
import fpwfsc.qacits.PID as PID
import ipdb

def printstatus(iteration=None,
                setpoint=None,
                center=None,
                pixcontrol=None,
                control=None):
    print("\n\n")
    print("Iteration: ", iteration)
    print("Setpoint: ", setpoint)
    print("Center: ", center)
    print("Error: ", setpoint-center)
    print("control in pixel coords: ", pixcontrol)
    print("Control in t/t = ", control)
    return



def run(camera=None, aosystem=None, config=None, configspec=None,
        my_deque=None, my_event=None, plotter=None, plot_signal=None,
        setpoint_queue=None):
    """
    Run the QACITS tracking loop
    
    Parameters:
    -----------
    camera : Camera object or 'Sim'
    aosystem : AO system object or 'Sim'
    config : Configuration object or path to config file
    configspec : Path to config spec file
    my_deque : Optional deque for data collection
    my_event : Threading event for stopping the loop
    plotter : DEPRECATED - use plot_signal instead
    plot_signal : PyQt signal for thread-safe plotting (emits: image, x_center, y_center, 
                  min_radius, max_radius, x_coords, y_coords, title)
    setpoint_queue : queue.Queue, optional
        Queue for receiving setpoint updates from GUI without resetting PID
    """
    if my_deque is None:
        my_deque = deque()

    if my_event is None:
        my_event = threading.Event()

    if camera == 'Sim' and aosystem == 'Sim':
        print("Using Simulator Mode")
        simmode = True
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

        hwsettings = sf.validate_config(str(hwconfig_path), str(hwspec_path))
        CSM      = fhw.FakeCoronagraphOpticalSystem(**hwsettings['SIMULATION']['OPTICAL_PARAMS'])
        AOSystem = fhw.FakeAODMSystem(OpticalModel=CSM, **hwsettings['SIMULATION']['AO_PARAMS'])
        Camera   = fhw.FakeDetector(opticalsystem=CSM, **hwsettings['SIMULATION']['CAMERA_PARAMS'])

    else:
        Camera = camera
        AOSystem = aosystem
        simmode = False
    settings = sf.validate_config(config, configspec)
    bgds = sf.setup_bgd_dict(hwsettings['CAMERA_CALIBRATION']['bgddir'])

    n_iter    = settings['EXECUTION']['N iterations']
    setpointx = settings['EXECUTION']['x setpoint']
    setpointy = settings['EXECUTION']['y setpoint']
    setpoint = np.array([setpointy, setpointx])
    inner_rad = settings['EXECUTION']['inner radius']
    outer_rad = settings['EXECUTION']['outer radius']
    # Use fixed values for spot search

    Kp = settings['PID']['Kp']
    Ki = settings['PID']['Ki']
    Kd = settings['PID']['Kd']
    output_limit = settings['PID']['output_limits']


    tt_gain = settings['AO']['tip tilt gain']
    tt_rot_deg  = settings['AO']['tip tilt angle (deg)']
    tt_flipx    = settings['AO']['tip tilt flip x']
    tt_flipy    = settings['AO']['tip tilt flip y']

    if settings['HITCHHIKER MODE']['hitchhike']:
        path_to_fits = settings['HITCHHIKER MODE']['imagedir']
        Hitch = fhw.Hitchhiker(imagedir=path_to_fits,
                               poll_interval=settings['HITCHHIKER MODE']['poll interval'],
                               timeout=settings['HITCHHIKER MODE']['timeout'])

    if simmode:
        #this should be about 5 pixels off
        initial_shape = AOSystem.get_dm_data()
        tt_control = dm.generate_tip_tilt(initial_shape.shape, tilt_x = 3e-6, tilt_y=-3e-6,
                                              dm_rotation=tt_rot_deg, flipx=False)
        current_shape = AOSystem.get_dm_data()
        AOSystem.set_dm_data(current_shape + tt_control)

    current_shape = AOSystem.get_dm_data()
    Controller = PID.PID(Kp=Kp,
                         Ki=Ki,
                         Kd=Kd,
                         output_limits=output_limit,
                         setpoint=np.array([0,0]))

    print(f"Starting control loop with initial setpoint: X={setpointx:.2f}, Y={setpointy:.2f}")

    for i in range(n_iter):
        # Check if stop event is set
        if my_event is not None and my_event.is_set():
            print('Stop event detected, stopping loop')
            break
        
        # Check for setpoint updates from queue
        if setpoint_queue is not None:
            try:
                # Non-blocking get - only update if there's a new setpoint
                new_setpoint = setpoint_queue.get_nowait()
                setpointy, setpointx = new_setpoint  # Queue has (y, x) order
                setpoint = np.array([setpointy, setpointx])
                print(f"✓ Setpoint updated mid-loop: X={setpointx:.2f}, Y={setpointy:.2f}")
                # Note: We DO NOT reset the PID controller - it maintains its state
            except queue.Empty:
                # No new setpoint, continue with current
                pass
        
        #drift = dm.generate_tip_tilt(initial_shape.shape, tilt_x = i*250e-9, tilt_y=400e-9,
        #                                  dm_rotation=tt_rot_deg, flipx=False)
        current_shape = AOSystem.get_dm_data()
        if settings['HITCHHIKER MODE']['hitchhike']:
            # Hitchhiker mode: wait for external process to generate images
            # Note: In sim mode, run qacits_camera_image_generator.py in a separate terminal
            # to generate images for this hitchhiker to find
            if simmode:
                # Don't auto-generate - wait for external file generator!
                pass
            try:
                data_speck_raw = Hitch.wait_for_next_image()
            except fhw.HitchhikerTimeoutError:
                print("Timeout waiting for image - exiting loop")
                break
        else:
            data_speck_raw = Camera.take_image()



        data_speck = sf.equalize_image(data_speck_raw, **bgds)
        #compute offset
        xs, ys, cropped = qf.crop_to_square(data_speck, cx=setpointx, cy = setpointy, size=outer_rad*4)
        xo, yo = qf.compute_quad_cell_flux(image=cropped, x_center=setpointx, y_center=setpointy, min_radius=inner_rad, max_radius=outer_rad,
               x_coords=xs, y_coords=ys)

        # Use signal for thread-safe plotting (new method)
        if plot_signal is not None:
            plottitle = '%.2f'%xo +', '+'%.2f'%yo
            plot_signal.emit(cropped, setpointx, setpointy, inner_rad, outer_rad, xs, ys, plottitle)
        # Legacy support for direct plotter (not thread-safe on macOS)
        elif plotter is not None:
            plottitle = '%.2f'%xo +', '+'%.2f'%yo
            plotter.update(image=cropped, x_center=setpointx, y_center=setpointy, min_radius=inner_rad, max_radius=outer_rad,
               x_coords=xs, y_coords=ys, title=plottitle)

        #pixcontrol = np.random.random(2)-0.5
        pixcontrol = Controller.iterate(np.array([xo, yo]))
        control = pixcontrol*tt_gain
        tt_control = dm.generate_tip_tilt(initial_shape.shape, tilt_x = control[0], tilt_y=control[1],
                                          dm_rotation=tt_rot_deg, flipx=False)
        current_shape = AOSystem.get_dm_data()
        AOSystem.set_dm_data(current_shape + tt_control)# + drift)
    
    print(f"Control loop ended after {i+1} iterations")
    
    # Only call execute if running standalone with a plotter
    if plotter is not None and plot_signal is None:
        plotter.execute()

if __name__ == "__main__":
    camera = 'Sim'
    aosystem = 'Sim'
    config = 'qacits_config.ini'
    configspec = 'qacits_config.spec'
    plotter = QacitsPlotter(figsize=(300*1.5,300*1.5))

    run(camera, aosystem, config=config,
                          configspec=configspec,
                          plotter=plotter)