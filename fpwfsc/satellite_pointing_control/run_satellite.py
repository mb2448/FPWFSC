import os, sys
import satellite_funcs as satfuncs

import numpy as np
import threading
from collections import deque

from fpwfsc.common import fake_hardware as fhw
import fpwfsc.common.support_functions as sf
import fpwfsc.common.dm as dm

import numpy as np
from PID import PID
from pathlib import Path
from satellite_plotter_qt import LiveSquarePlotter

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

def apply_waffle(aosystem=None, waffle_amplitude=None):
    initial_shape = aosystem.get_dm_data()
    waffle = dm.generate_waffle(initial_shape, amplitude=waffle_amplitude)
    aosystem.set_dm_data(initial_shape + waffle)
    print(waffle)
    return True

def remove_waffle(aosystem=None):
    initial_shape = aosystem.get_dm_data()
    withoutwaffle = remove_waffle(initial_shape)
    aosystem.set_dm_data(withoutwaffle)
    return True

def run(camera=None, aosystem=None, config=None, configspec=None,
        my_deque=None, my_event=None, plotter=None):
    if my_deque is None:
        my_deque = deque()

    if my_event is None:
        my_event = threading.Event()
    
    if camera == 'Sim' and aosystem == 'Sim':
        print("Using Simulator Mode")
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
    
    settings = sf.validate_config(config, configspec)
    bgds = sf.setup_bgd_dict(hwsettings['CAMERA_CALIBRATION']['bgddir'])
    
    n_iter    = settings['EXECUTION']['N iterations']
    setpointx = settings['EXECUTION']['x setpoint']
    setpointy = settings['EXECUTION']['y setpoint']
    setpoint = np.array([setpointy, setpointx])
    centerguess = setpoint.copy()
    radius      = settings['EXECUTION']['spot search radius (pix)']
    radtol      = settings['EXECUTION']['radius tolerance (pix)']
    
    Kp = settings['PID']['Kp']
    Ki = settings['PID']['Ki']
    Kd = settings['PID']['Kd']
    output_limit = settings['PID']['output_limits']

    waffleamp = settings['AO']['waffle mode amplitude']
    tt_gain = settings['AO']['tip tilt gain']
    tt_rot_deg  = settings['AO']['tip tilt angle (deg)']
    tt_flipx    = settings['AO']['tip tilt flip x']
    tt_flipy    = settings['AO']['tip tilt flip y']

    initial_shape = AOSystem.get_dm_data()
    data_nospeck_raw = Camera.take_image()
    data_nospeck = sf.equalize_image(data_nospeck_raw, **bgds)

    waffle = dm.generate_waffle(initial_shape, amplitude = waffleamp)
    AOSystem.set_dm_data(initial_shape + waffle)
    data_speck_raw = Camera.take_image()
    data_speck = sf.equalize_image(data_speck_raw, **bgds)
    #this should be about 5 pixels off
    tt_control = dm.generate_tip_tilt(initial_shape.shape, tilt_x = -1000e-9, tilt_y=0,
                                          dm_rotation=tt_rot_deg, flipx=False)
    current_shape = AOSystem.get_dm_data()
    AOSystem.set_dm_data(current_shape + tt_control)
    PIDloop = PID(Kp=Kp, Ki=Ki, Kd = Kd, setpoint=setpoint, output_limits=(-output_limit, output_limit))
    current_shape = AOSystem.get_dm_data()
    for i in range(n_iter):
        # Check if stop event is set
        if my_event is not None and my_event.is_set():
            print('Stop event detected, stopping loop')
            break
        #drift = dm.generate_tip_tilt(initial_shape.shape, tilt_x = 4*250e-9, tilt_y=400e-9,
        #                                  dm_rotation=35, flipx=False)
        current_shape = AOSystem.get_dm_data()
        data_speck_raw = Camera.take_image()
        data_speck = sf.equalize_image(data_speck_raw, **bgds)
        try:
            points = satfuncs.find_spots_in_annulus(data_speck, centerguess, radius, tol=radtol,
                                                 min_area=5, max_area=150)
            center, side, theta, _ = satfuncs.fit_square_and_center(points)
        except Exception as e:
            print(f"Error in spot detection: {e}")
            print("Skipping iteration ", i)
            continue  

        if plotter is not None:
            plotter.update(image=data_speck,
                           setpoint=setpoint,
                           center=center,
                           side=side,
                           theta=theta,
                           points=points,
                           radius=radius,
                           radtol=radtol,
                           centerguess=centerguess,
                           title='%.2f'%center[0]+', '+'%.2f'%center[1])

        #if np.random.random()<0.02:
        #    setpoint = setpoint + np.random.randint(-5, 6, size= 2)
        #    PIDloop.setpoint = setpoint
        centerguess = center
        pixcontrol = 1*np.array(PIDloop(center))
        control = pixcontrol*tt_gain
        printstatus(iteration=i,
                    setpoint=setpoint,
                    center=center,
                    pixcontrol=pixcontrol,
                    control=control)
        tt_control = dm.generate_tip_tilt(initial_shape.shape, tilt_x = control[1], tilt_y=control[0],
                                          dm_rotation=tt_rot_deg, flipx=False)
        print("\n\n")
        current_shape = AOSystem.get_dm_data()
        AOSystem.set_dm_data(current_shape + tt_control)# + drift)
    if plotter is not None:
        plotter.execute()

if __name__ == "__main__":
    camera = 'Sim'
    aosystem = 'Sim'
    config = 'satellite_config.ini'
    configspec = 'satellite_config.spec'
    plotter = LiveSquarePlotter(figsize=(300*1.5,600*1.5))
    
    run(camera, aosystem, config=config,
                          configspec=configspec, 
                          plotter=plotter)
