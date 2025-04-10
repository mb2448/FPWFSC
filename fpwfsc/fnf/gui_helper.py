# gui_helper.py
from ..common import bench_hardware as hw

valid_instruments = ['Sim', 'NIRC2', 'Palila', 'Vampires']

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
        }
    },
    "FF_SETTINGS": {
        "xcen": {
            "help": "The x position of the target psf, in instrument coordinates (eg, pixel 350)",
            "expert": False
        },
        "ycen": {
            "help": "The x position of the target psf, in instrument coordinates (eg, pixel 350)",
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
        }
    },
    "IO": {
        "save path": {
            "help": "Path to save output files",
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
    elif instrumentname == 'NIRC2':
        return hw.NIRC2(**camargs), hw.KeckAO(**aoargs)
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

#def get_expert_options(section):
#    """Get a list of expert options for a given section."""
#    return [key for key, value in config_info.get(section, {}).items() if value.get("expert", False)]

#def get_all_options(section):
#    """Get a list of all options for a given section."""
#    return list(config_info.get(section, {}).keys())
