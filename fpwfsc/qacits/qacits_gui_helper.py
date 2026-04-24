# qacits_gui_helper.py

# Define valid instruments
valid_instruments = ['Sim', 'NIRC2']

# Configuration information dictionary
config_info = {
    "EXECUTION": {
        "plot": {
            "help": "Enable live image plotting during the loop",
            "expert": False
        },
        "log": {
            "help": "Enable logging of telemetry to a text file each run",
            "expert": False
        },
        "logdir": {
            "help": "Directory to write log files into",
            "expert": False,
            "directory": True
        },
        "N iterations": {
            "help": "Number of control loop iterations to run",
            "expert": False
        },
        "x setpoint": {
            "help": "X pixel coordinate where the star is expected on the detector",
            "expert": False
        },
        "y setpoint": {
            "help": "Y pixel coordinate where the star is expected on the detector",
            "expert": False
        },
        "inner radius": {
            "help": "Inner radius of the quad cell annulus in pixels (0 = full circle)",
            "expert": True
        },
        "outer radius": {
            "help": "Outer radius of the quad cell annulus in pixels. Also sets the crop size (2*outer_rad+1).",
            "expert": True
        },
    },
    "CAMERA CALIBRATION": {
        "background file": {
            "help": "Path to background/dark FITS file. If blank, background is estimated from the border rows/columns of each frame.",
            "expert": True,
            "file": True
        },
        "masterflat file": {
            "help": "Path to mean-normalized master flat FITS file. If blank, no flat correction is applied.",
            "expert": True,
            "file": True
        },
        "badpix file": {
            "help": "Path to bad pixel map FITS file (nonzero = bad). If blank, no bad pixel correction is applied.",
            "expert": True,
            "file": True
        },
    },
    "HITCHHIKER MODE": {
        "hitchhike": {
            "help": "Enable hitchhiker mode: read images from a directory instead of controlling the camera",
            "expert": False
        },
        "imagedir": {
            "help": "Directory to watch for new FITS files in hitchhiker mode",
            "expert": True,
            "directory": True
        },
        "poll interval": {
            "help": "How often (seconds) to check the directory for new files",
            "expert": True
        },
        "timeout": {
            "help": "Exit the loop if no new file appears within this many seconds",
            "expert": True
        },
    },
    "PID": {
        "x centroid offset": {
            "help": "Quad cell X target in centroid units (0 = centered on setpoint pixel). Range -1 to 1.",
            "expert": False
        },
        "y centroid offset": {
            "help": "Quad cell Y target in centroid units (0 = centered on setpoint pixel). Range -1 to 1.",
            "expert": False
        },
        "Kp": {
            "help": "Proportional gain. Start with 0.3-0.5 and increase until convergence without oscillation.",
            "expert": True
        },
        "Ki": {
            "help": "Integral gain. Set to 0 initially; add small values to eliminate steady-state offset.",
            "expert": True
        },
        "Kd": {
            "help": "Derivative gain. Usually 0; can reduce overshoot if Kp is high.",
            "expert": True
        },
        "output_limits": {
            "help": "Maximum PID output magnitude in centroid units. Symmetric +/- clamp. Prevents large corrections from noisy measurements.",
            "expert": True
        }
    },
    "AO": {
        "tip tilt gain": {
            "help": "Conversion factor from centroid units to AO tip-tilt command units. Sign and magnitude depend on the instrument.",
            "expert": True
        },
        "tip tilt angle (deg)": {
            "help": "Rotation angle between detector axes and AO axes in degrees. Calibrate per instrument.",
            "expert": True
        },
        "tip tilt flip x": {
            "help": "Flip the X correction direction (negate X before sending to AO)",
            "expert": True
        },
        "tip tilt flip y": {
            "help": "Flip the Y correction direction (negate Y before sending to AO)",
            "expert": True
        }
    },
}

def load_instruments(instrumentname, camargs={}, aoargs={}):
    """
    Load camera and AO system for the selected instrument.

    Returns (camera, aosystem) where each is either a string sentinel
    ('Sim') or a hardware wrapper object.
    """
    if instrumentname == 'Sim':
        return 'Sim', 'Sim'
    elif instrumentname == 'NIRC2':
        from fpwfsc.qacits import qacits_hardware
        return qacits_hardware.NIRC2Alias(), qacits_hardware.K2AOAlias()
    else:
        raise ValueError(f"Invalid instrument name: {instrumentname}")

def get_help_message(section, key):
    """Retrieve the help message for a given section and key."""
    return config_info.get(section, {}).get(key, {}).get("help", "No help available")

def is_expert_option(section, key):
    """Check if a given option is an expert option."""
    return config_info.get(section, {}).get(key, {}).get("expert", False)

def is_directory_option(section, key):
    """Check if a given option requires a directory selection."""
    return config_info.get(section, {}).get(key, {}).get("directory", False)

def is_file_option(section, key):
    """Check if a given option requires a file selection."""
    return config_info.get(section, {}).get(key, {}).get("file", False)
