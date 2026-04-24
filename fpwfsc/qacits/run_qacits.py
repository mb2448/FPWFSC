import numpy as np
import threading
import queue  # For thread-safe setpoint updates

from fpwfsc.common import fake_hardware as fhw
import fpwfsc.common.support_functions as sf
import fpwfsc.common.dm as dm

from pathlib import Path
import fpwfsc.qacits.qacits_funcs as qf
import fpwfsc.qacits.PID as PID

def printstatus(iteration=None,
                setpoint=None,
                quad_cell=None,
                centroid_setpoint=None,
                correction=None,
                control=None):
    print("\n\n")
    print("Iteration: ", iteration)
    print("Setpoint (px): ", setpoint)
    print("Quad cell (x, y): ", quad_cell)
    print("PID setpoint (centroid): ", centroid_setpoint)
    print("PID output (centroid): ", correction)
    print("Control (t/t): ", control)
    return

def run(camera=None, aosystem=None, config=None, configspec=None,
        my_event=None, plotter=None, plot_signal=None,
        setpoint_queue=None, centroid_offset_queue=None,
        centroid_offset_feedback=None):
    """
    Run the QACITS tracking loop

    Parameters:
    -----------
    camera : Camera object or 'Sim'
    aosystem : AO system object or 'Sim'
    config : Configuration object or path to config file
    configspec : Path to config spec file
    my_event : Threading event for stopping the loop
    plotter : DEPRECATED - use plot_signal instead
    plot_signal : PyQt signal for thread-safe plotting (emits: image, x_center, y_center,
                  min_radius, max_radius, x_coords, y_coords, title)
    setpoint_queue : queue.Queue, optional
        Queue for receiving setpoint updates from GUI without resetting PID
    centroid_offset_queue : queue.Queue, optional
        Queue for receiving centroid offset updates. Send (x, y) to set a
        specific offset, or None to capture the current quad-cell reading.
    centroid_offset_feedback : queue.Queue, optional
        Queue for sending back applied centroid offsets to the GUI.
    """
    if my_event is None:
        my_event = threading.Event()

    if camera == 'Sim' and aosystem == 'Sim':
        print("Using Simulator Mode")
        simmode = True
        script_dir = Path(__file__).parent.absolute()
        hwconfig_path = script_dir.parent / "sim" / "sim_config.ini"
        hwspec_path = script_dir.parent / "sim" / "sim_config.spec"

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

    bgds = {
        'bkgd':       sf.load_fits_or_none(settings['CAMERA CALIBRATION']['background file']),
        'masterflat': sf.load_fits_or_none(settings['CAMERA CALIBRATION']['masterflat file']),
        'badpix':     sf.load_fits_or_none(settings['CAMERA CALIBRATION']['badpix file']),
    }
    print(f"Background file: {settings['CAMERA CALIBRATION']['background file'] or '(none)'}")
    print(f"Masterflat file: {settings['CAMERA CALIBRATION']['masterflat file'] or '(none)'}")
    print(f"Bad pixel file:  {settings['CAMERA CALIBRATION']['badpix file'] or '(none)'}")

    n_iter    = settings['EXECUTION']['N iterations']
    setpointx = settings['EXECUTION']['x setpoint']
    setpointy = settings['EXECUTION']['y setpoint']
    setpoint = np.array([setpointy, setpointx])
    inner_rad = settings['EXECUTION']['inner radius']
    outer_rad = settings['EXECUTION']['outer radius']
    if outer_rad <= inner_rad:
        raise ValueError(f"outer radius ({outer_rad}) must be strictly greater than inner radius ({inner_rad})")

    centroid_offset_x = settings['PID']['x centroid offset']
    centroid_offset_y = settings['PID']['y centroid offset']
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
        # Inject a fixed physical tilt so the loop has something to correct.
        # This is independent of the controller's rotation/flip calibration.
        AOSystem.offset_tiptilt(3e-6, -3e-6)

    centroid_setpoint = np.array([centroid_offset_x, centroid_offset_y])
    Controller = PID.PID(Kp=Kp,
                         Ki=Ki,
                         Kd=Kd,
                         output_limits=output_limit,
                         setpoint=centroid_setpoint)

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


        #Check if bgd files are the right size
        if i == 0:
            for key, arr in bgds.items():
                if arr is not None and arr.shape != data_speck_raw.shape:
                    raise ValueError(f"Calibration file shape mismatch: '{key}' is {arr.shape}, "
                                     f"camera image is {data_speck_raw.shape}")

        data_speck = sf.equalize_image(data_speck_raw, **bgds)
        #compute offset
        xs, ys, cropped = qf.crop_to_square(data_speck, cx=setpointx, cy=setpointy, size=outer_rad*2 + 1)
        xo, yo = qf.compute_quad_cell_flux(image=cropped, x_center=setpointx, y_center=setpointy, min_radius=inner_rad, max_radius=outer_rad,
               x_coords=xs, y_coords=ys)

        # Check for centroid offset updates
        if centroid_offset_queue is not None:
            try:
                new_offset = centroid_offset_queue.get_nowait()
                if new_offset is None:
                    # Capture current reading
                    centroid_setpoint = np.array([xo, yo])
                else:
                    centroid_setpoint = np.array(new_offset)
                Controller.update_setpoint(centroid_setpoint)
                print(f"Centroid offset set to: x={centroid_setpoint[0]:.4f}, y={centroid_setpoint[1]:.4f}")
                if centroid_offset_feedback is not None:
                    centroid_offset_feedback.put((centroid_setpoint[0], centroid_setpoint[1]))
            except queue.Empty:
                pass

        # Use signal for thread-safe plotting (new method)
        if plot_signal is not None:
            plottitle = '%.4f'%xo +', '+'%.4f'%yo
            plot_signal.emit(cropped, setpointx, setpointy, inner_rad, outer_rad, xs, ys, plottitle)
        # Legacy support for direct plotter (not thread-safe on macOS)
        elif plotter is not None:
            plottitle = '%.4f'%xo +', '+'%.4f'%yo
            plotter.update(image=cropped, x_center=setpointx, y_center=setpointy, min_radius=inner_rad, max_radius=outer_rad,
               x_coords=xs, y_coords=ys, title=plottitle)

        #correction = np.random.random(2)-0.5
        correction = Controller.iterate(np.array([xo, yo]))
        control = correction*tt_gain

        printstatus(iteration=i,
                    setpoint=setpoint,
                    quad_cell=np.array([xo, yo]),
                    centroid_setpoint=Controller.setpoint,
                    correction=correction,
                    control=control)
        
        x_ao, y_ao = dm.rotate_flip_tt(control[0], control[1], rot_deg=tt_rot_deg,
                                       flipx=tt_flipx, flipy=tt_flipy)
        AOSystem.offset_tiptilt(x_ao, y_ao)

    print(f"Control loop ended after {i+1 if n_iter > 0 else 0} iterations")
    
    # Only call execute if running standalone with a plotter
    if plotter is not None and plot_signal is None:
        plotter.execute()

if __name__ == "__main__":
    from fpwfsc.qacits.qacits_plotter_qt import QacitsPlotter
    camera = 'Sim'
    aosystem = 'Sim'
    config = 'qacits_config.ini'
    configspec = 'qacits_config.spec'
    plotter = QacitsPlotter(figsize=(300*1.5,300*1.5))

    run(camera, aosystem, config=config,
                          configspec=configspec,
                          plotter=plotter)