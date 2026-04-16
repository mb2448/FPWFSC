#!/usr/bin/env python3
"""
Simulated image generator.

Two modes:
  1. Show: take one image and display it (verify sim config works).
  2. Stream: write FITS files to a directory at a regular interval
     (feed hitchhiker mode in qacits / fnf / satellite).

Usage:
    # Show a single image (diagnostic)
    python -m fpwfsc.sim.sim_image_generator --show

    # Stream FITS to a directory
    python -m fpwfsc.sim.sim_image_generator /path/to/dir
    python -m fpwfsc.sim.sim_image_generator /path/to/dir --interval 2.0
    python -m fpwfsc.sim.sim_image_generator /path/to/dir --count 50
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import astropy.io.fits as pf

import fpwfsc.common.support_functions as sf
import fpwfsc.common.fake_hardware as fhw
import fpwfsc.common.dm as dm


def _load_sim():
    """Build the sim optical system + DM + camera from sim_config."""
    sim_dir = Path(__file__).parent.absolute()
    hwconfig = sim_dir / "sim_config.ini"
    hwspec = sim_dir / "sim_config.spec"
    if not hwconfig.exists():
        raise FileNotFoundError(f"Sim config not found: {hwconfig}")
    hw = sf.validate_config(str(hwconfig), str(hwspec))

    opt = hw['SIMULATION']['OPTICAL_PARAMS']
    cam = hw['SIMULATION']['CAMERA_PARAMS']
    ao = hw['SIMULATION']['AO_PARAMS']

    CSM = fhw.FakeCoronagraphOpticalSystem(**opt)
    AOSystem = fhw.FakeAODMSystem(OpticalModel=CSM, **ao)
    Camera = fhw.FakeDetector(opticalsystem=CSM, **cam)

    return CSM, AOSystem, Camera, cam, opt


def show():
    """Take one image and display it."""
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    _, _, Camera, cam, opt = _load_sim()
    img = Camera.take_image()

    print(f"N pix focal (hcipy grid): {opt['N pix focal']}")
    print(f"Canvas size: {cam['xsize']} x {cam['ysize']}")
    print(f"Field center: ({cam['field_center_x']}, {cam['field_center_y']})")
    print(f"Output shape: {img.shape}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    vmin, vmax = np.percentile(img, [1, 99])
    ax1.imshow(img, origin='lower', vmin=vmin, vmax=vmax)
    ax1.set_title(f"Full frame {img.shape[1]}x{img.shape[0]}  (1-99% stretch)")
    ax1.axhline(cam['field_center_y'], color='r', ls='--', alpha=0.5)
    ax1.axvline(cam['field_center_x'], color='r', ls='--', alpha=0.5)

    hw = 50
    cx, cy = cam['field_center_x'], cam['field_center_y']
    cutout = img[max(0, cy-hw):cy+hw, max(0, cx-hw):cx+hw]
    vmin2, vmax2 = np.percentile(cutout, [1, 99])
    ax2.imshow(cutout, origin='lower', vmin=vmin2, vmax=vmax2)
    ax2.set_title(f"Cutout around ({cx}, {cy})")

    plt.tight_layout()
    plt.show()


def stream(output_dir, interval=1.0, count=None, drift=True):
    """Write FITS images to output_dir at regular intervals."""
    image_dir = Path(output_dir).absolute()
    os.makedirs(image_dir, exist_ok=True)
    if not os.access(image_dir, os.W_OK):
        raise PermissionError(f"Cannot write to directory: {image_dir}")

    _, AOSystem, Camera, _, _ = _load_sim()

    # Add an initial tip-tilt offset so the PSF is off-center
    AOSystem.offset_tiptilt(1e-6, -1e-6)

    print(f"Streaming to: {image_dir}")
    print(f"Interval: {interval}s, Count: {'infinite' if count is None else count}")
    print("Press Ctrl+C to stop\n")

    i = 0
    try:
        while count is None or i < count:
            t0 = time.time()
            data = Camera.take_image()

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = image_dir / f"sim_{timestamp}_{i:04d}.fits"
            pf.writeto(filepath, data, overwrite=True)
            print(f"[{i}] wrote {filepath.name}")

            if drift:
                AOSystem.offset_tiptilt(np.random.randn()*200e-9,
                                        np.random.randn()*200e-9)

            i += 1
            elapsed = time.time() - t0
            if count is None or i < count:
                time.sleep(max(0.0, interval - elapsed))

    except KeyboardInterrupt:
        print(f"\nStopped. {i} images written.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Directory to write FITS files. Omit for --show mode.")
    parser.add_argument("--show", action="store_true",
                        help="Take one image and display it (diagnostic mode)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between images in stream mode (default: 1.0)")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of images to generate (default: infinite)")
    parser.add_argument("--no-drift", action="store_true",
                        help="Disable random tip-tilt drift between frames")
    args = parser.parse_args()

    if args.show or args.output_dir is None:
        show()
    else:
        stream(args.output_dir, interval=args.interval,
               count=args.count, drift=not args.no_drift)


if __name__ == "__main__":
    main()
