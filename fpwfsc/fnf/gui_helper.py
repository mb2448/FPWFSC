# gui_helper.py
from ..common import bench_hardware as hw
import sys
# sys.path.insert(0, '/usr/local/home/cguthery/PyAO/')
# from guis.fast_and_furious import hardware
valid_instruments = ['Sim', 'NIRC2', 'Palila', 'Vampires','OSIRIS']

config_info = {
    "LOOP_SETTINGS": {
        "Plot": {
            "help": "Enable plotting (SLOW)",
            "expert":False
        },
        "N iter": {
            "help": "Number of iterations for the loop",
            "expert": False
        },
        "gain": {
            "help": "Gain factor for the control loop",
            "expert": False
        },
        "leak factor": {
            "help": "Leakage factor to prevent integrator wind-up",
            "expert": True
        },
        "Used mode basis": {
            "help": "Basis used for mode decomposition",
            "expert": False
        },
        "Number of modes": {
            "help": "Number of modes to correct",
            "expert": False
        },
        "N images averaged": {
            "help": "Number of images to average from the camera",
            "expert": True
        },
        "control even modes": {
            "help": "Enable control of even modes",
            "expert": True
        },
        "control odd modes": {
            "help": "Enable control of odd modes",
            "expert": True
        },
        "dm command boost": {
            "help": "Boost factor for DM commands",
            "expert": True
        }
    },
    "MODELLING": {
        "wavelength (m)": {
            "help": "Wavelength of light in meters",
            "expert": False
        },
        "pixel scale (mas/pix)": {
            "help": "Pixel scale in milli-arcseconds per pixel",
            "expert": True
        },
        "N pix pupil": {
            "help": "Number of pixels across the pupil",
            "expert": True
        },
        "N pix focal": {
            "help": "Number of pixels in the focal plane",
            "expert": False
        },
        "aperture": {
            "help": "Type of aperture used",
            "expert": False
        },
        "grid diameter (m)": {
            "help": "Diameter of the grid used to generate the pupil",
            "expert": True
        },
        "rotation angle aperture (deg)": {
            "help": "Rotation angle of the aperture in degrees",
            "expert": True
        },
        "rotation angle dm (deg)": {
            "help": "Rotation angle of the deformable mirror in degrees",
            "expert": True
        },
        "flip_x": {
            "help": "Flip the x-axis of the pupil",
            "expert": True
        },
        "flip_y": {
            "help": "Flip the y-axis of the pupil",
            "expert": True
        },
        "ref PSF oversampling factor": {
            "help": "Oversampling factor for the reference PSF",
            "expert": True
        },
        "test time": {
            "help": "Daytime or nighttime observation",
            "expert": True
        }
    },
    "FF_SETTINGS": {
        "xcen": {
            "help": """The x position of the target psf, in instrument coordinates (eg, pixel 350).
                        does not need to be accurate to more than a few pixels.""",
            "expert": False
        },
        "ycen": {
            "help": """The x position of the target psf, in instrument coordinates (eg, pixel 350).
                        does not need to be accurate to more than a few pixels.""",
            "expert": False
        },
        "Apply smooth filter": {
            "help": "Apply a smoothing filter to the wavefront",
            "expert": True
        },
        "SNR cutoff": {
            "help": "Signal-to-noise ratio cutoff for mode filtering",
            "expert": True
        },
        "epsilon": {
            "help": "Small value to prevent division by zero",
            "expert": True
        },
        "auto_background": {
            "help": "Automatically determine background level",
            "expert": False
        },
        "hitchhiker_mode": {
            "help": "Use science data for FnF",
            "expert": False
        },
        "hitchhiker_path": {
            "help": "Directory to look for new science data",
            "expert": True
        },
        "save_log": {
            "help": "Save all measuremnets",
            "expert": False
        },
        "log_path": {
            "help": "Directory o save the log",
            "expert": True
        }
    },
    "IO": {
        "save path": {
            "help": "Path to save output files",
            "directory":True,
            "expert": False
        },
        "plot dm command": {
            "help": "Enable plotting of DM commands",
            "expert": True
        }
    },
    "SIMULATION": {
        "run_sim": {
            "help": "Enable simulation mode",
            "expert": False
        },
        "flux": {
            "help": "Photon flux value",
            "expert": True
        },
        "exptime": {
            "help": "Exposure time in seconds",
            "expert": True
        },
        "rms_wfe": {
            "help": "RMS wavefront error",
            "expert": True
        },
        "seed": {
            "help": "The random seed for the error",
            "expert": True
        }
    }
}

def load_instruments(instrumentname, camargs={}, aoargs={}):
    if instrumentname == 'Sim':
        return 'Sim','Sim'
    #XXX changed to alias function as the original one no longer exis in bench_hardware
    elif instrumentname == 'NIRC2':
        return hardware.NIRC2(**camargs), hw.ClosedAOSystemAlias(**aoargs)
    elif instrumentname == 'OSIRIS':
        return hardware.OSIRIS(**camargs), hw.ClosedAOSystemAlias(**aoargs)

    elif instrumentname == 'Palila':
        return hw.Palila(**camargs), hw.SCEXAO(**aoargs)
    elif instrumentname == 'Vampires':
        return hw.Vampires(**camargs), hw.SCEXAO(**aoargs)
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

#def get_expert_options(section):
#    """Get a list of expert options for a given section."""
#    return [key for key, value in config_info.get(section, {}).items() if value.get("expert", False)]

#def get_all_options(section):
#    """Get a list of all options for a given section."""
#    return list(config_info.get(section, {}).keys())
