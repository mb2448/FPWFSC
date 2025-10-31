import os, sys

import numpy as np

from fpwfsc.common import fake_hardware as fhw
import fpwfsc.common.support_functions as sf
import fpwfsc.common.dm as dm

from fpwfsc.satellite_pointing_control.PID import PID
from pathlib import Path



if __name__ == "__main__":
    script_dir = Path(__file__).parent.absolute()
    hwconfig_path = script_dir/ "qacits_config.ini"
    hwspec_path = script_dir/ "qacits_config.spec"

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
