#!/usr/bin/env python3
"""
QACITS File Generator for Hitchhiker Mode Testing

This script generates simulated camera images at regular intervals
for testing the hitchhiker mode in qacits_GUI.py

Usage:
    python qacits_file_generator.py

    # Or with custom interval:
    python qacits_file_generator.py --interval 2.0

    # Or with custom number of images:
    python qacits_file_generator.py --count 50
"""

import os
import sys
import time
import argparse
from pathlib import Path

from fpwfsc.common import fake_hardware as fhw
import fpwfsc.common.support_functions as sf
import fpwfsc.common.dm as dm

def generate_images(output_dir, interval=1.0, count=None):
    """
    Generate simulated camera images at regular intervals

    Parameters:
    -----------
    output_dir : str or Path
        Directory to save images to
    interval : float
        Time between images in seconds (default 1.0)
    count : int, optional
        Number of images to generate. If None, runs forever
    """
    image_dir = Path(output_dir).absolute()
    print(f"Output directory: {image_dir}")

    # Make sure directory exists
    os.makedirs(image_dir, exist_ok=True)

    # Verify we can write to it
    if not os.access(image_dir, os.W_OK):
        raise PermissionError(f"Cannot write to directory: {image_dir}")

    # Load hardware config for simulator
    script_dir = Path(__file__).parent.absolute()
    hwconfig_path = script_dir.parent / "san" / "sn_config.ini"
    hwspec_path = script_dir.parent / "san" / "sn_config.spec"

    print(f"Loading hardware config from: {hwconfig_path}")

    if not hwconfig_path.exists():
        print(f"WARNING: Hardware config not found at {hwconfig_path}")
        print("Trying alternative location...")
        # Try current directory
        hwconfig_path = Path("../san/sn_config.ini").absolute()
        hwspec_path = Path("../san/sn_config.spec").absolute()

        if not hwconfig_path.exists():
            raise FileNotFoundError(
                f"Hardware config not found. Tried:\n"
                f"  1. {script_dir.parent / 'san' / 'sn_config.ini'}\n"
                f"  2. {hwconfig_path}\n"
                f"Make sure you're running from the qacits directory."
            )

    if not hwspec_path.exists():
        raise FileNotFoundError(f"Hardware spec not found: {hwspec_path}")

    hwsettings = sf.validate_config(str(hwconfig_path), str(hwspec_path))

    # Initialize simulator
    print("Initializing optical system simulator...")
    CSM = fhw.FakeCoronagraphOpticalSystem(**hwsettings['SIMULATION']['OPTICAL_PARAMS'])
    AOSystem = fhw.FakeAODMSystem(OpticalModel=CSM, **hwsettings['SIMULATION']['AO_PARAMS'])

    # Override the save directory to use our specified output directory
    camera_params = hwsettings['SIMULATION']['CAMERA_PARAMS'].copy()
    camera_params['output_directory'] = str(image_dir)
    Camera = fhw.FakeDetector(opticalsystem=CSM, **camera_params)

    print(f"Camera configured to save to: {image_dir}")

    # Add some initial tip-tilt offset to make it more interesting
    print("Adding initial tip-tilt offset...")
    initial_shape = AOSystem.get_dm_data()
    # Use default rotation angle
    tt_rot_deg = 0  # degrees
    tt_control = dm.generate_tip_tilt(initial_shape.shape,
                                      tilt_x=1e-6,
                                      tilt_y=-1e-6,
                                      dm_rotation=tt_rot_deg,
                                      flipx=False)
    current_shape = AOSystem.get_dm_data()
    AOSystem.set_dm_data(current_shape + tt_control)

    print("\n" + "="*60)
    print("File Generator Started!")
    print(f"Output Directory: {image_dir}")
    print(f"Interval: {interval} seconds")
    print(f"Count: {'Infinite' if count is None else count}")
    print("="*60 + "\n")
    print("Press Ctrl+C to stop\n")

    i = 0
    try:
        while True:
            # Check if we've reached the count limit
            if count is not None and i >= count:
                print(f"\nReached target count of {count} images. Exiting.")
                break

            # Generate image
            print(f"[{i+1}] Generating image...", end=' ', flush=True)
            start_time = time.time()

            # Take image (this saves it to the directory automatically)
            data = Camera.take_image()

            elapsed = time.time() - start_time
            print(f"Done in {elapsed:.3f}s")

            # Add small random drift every 
            import numpy as np
            drift = dm.generate_tip_tilt(initial_shape.shape,
                                         tilt_x=np.random.randn()*200e-9,
                                         tilt_y=np.random.randn()*200e-9,
                                         dm_rotation=tt_rot_deg,
                                         flipx=False)
            current_shape = AOSystem.get_dm_data()
            AOSystem.set_dm_data(current_shape + drift)
            print(f"    Added random drift")

            i += 1

            # Wait for the specified interval
            if count is None or i < count:
                time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print(f"Stopped by user. Generated {i} images.")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Generate simulated camera images for QACITS hitchhiker mode testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate images to specific directory, every 1 second
  python qacits_file_generator.py /path/to/output/directory

  # Generate images every 2 seconds
  python qacits_file_generator.py /path/to/output/directory --interval 2.0

  # Generate exactly 50 images
  python qacits_file_generator.py /path/to/output/directory --count 50

  # Fast generation (every 0.5 seconds)
  python qacits_file_generator.py /path/to/output/directory --interval 0.5

  # Using relative path
  python qacits_file_generator.py ../dummy_camera_directory --interval 1.5
        """
    )

    parser.add_argument('directory', type=str,
                       help='Output directory for generated images')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Time between images in seconds (default: 1.0)')
    parser.add_argument('--count', type=int, default=None,
                       help='Number of images to generate (default: infinite)')

    args = parser.parse_args()

    # Validate arguments
    if args.interval <= 0:
        print("Error: interval must be positive")
        sys.exit(1)

    if args.count is not None and args.count <= 0:
        print("Error: count must be positive")
        sys.exit(1)

    # Run the generator
    try:
        generate_images(output_dir=args.directory,
                       interval=args.interval,
                       count=args.count)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()