# satellite_gui_helper.py

# Define valid instruments
valid_instruments = ['Sim']

# Configuration information dictionary
config_info = {
    "HITCHHIKER MODE": {
        "hitchhike": {
            "help": "Enable hitchhiker mode to look for files in a directory",
            "expert": True
        },
        "imagedir": {
            "help": "Directory path for image files in hitchhiker mode",
            "expert": True,
            "directory": True
        }
    },
    "EXECUTION": {
        "plot": {
            "help": "Enable plotting",
            "expert": False
        },
        "n_iter": {
            "help": "Number of iterations for the algorithm",
            "expert": False
        },
        "x setpoint": {
            "help": "X-coordinate of the setpoint",
            "expert": False
        },
        "y setpoint": {
            "help": "Y-coordinate of the setpoint",
            "expert": False
        },
        "spot search radius (pix)": {
            "help": "Radius in pixels to search for spots",
            "expert": True
        },
        "radius tolerance (pix)": {
            "help": "Tolerance for the search radius",
            "expert": True
        }
    },
    "AO": {
        "default_waffle_amplitude": {
            "help": "Amplitude for waffle pattern",
            "expert": False
        },
        "tip tilt gain": {
            "help": "Gain to convert from pixels to TT_OFFSET commands",
            "expert": True
        },
        "tip tilt angle (deg)": {
            "help": "Rotation angle for tip-tilt in degrees",
            "expert": True
        },
        "tip tilt flip x": {
            "help": "Flip the X-axis for tip-tilt",
            "expert": True
        },
        "tip tilt flip y": {
            "help": "Flip the Y-axis for tip-tilt",
            "expert": True
        }
    },
    "PID": {
        "Kp": {
            "help": "Proportional gain for PID control",
            "expert": True
        },
        "Ki": {
            "help": "Integral gain for PID control",
            "expert": True
        },
        "Kd": {
            "help": "Derivative gain for PID control",
            "expert": True
        },
        "output_limits": {
            "help": "Maximum output from PID controller in pixels",
            "expert": True
        }
    }
}

def load_instruments(instrumentname, camargs={}, aoargs={}):
    """
    Load instruments based on the selected name
    For now, only simulation mode is supported
    """
    if instrumentname == 'Sim':
        return 'Sim', 'Sim'
    else:
        raise ValueError("Invalid instrument name")

def get_help_message(section, key):
    """Retrieve the help message for a given section and key."""
    return config_info.get(section, {}).get(key, {}).get("help", "No help available")

def is_expert_option(section, key):
    """Check if a given option is an expert option."""
    return config_info.get(section, {}).get(key, {}).get("expert", False)

def is_directory_option(section, key):
    """Check if a given option requires a directory selection."""
    return config_info.get(section, {}).get(key, {}).get("directory", False)
