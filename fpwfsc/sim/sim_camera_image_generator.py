"""
Quick diagnostic: load sim_config, build the fake camera, take one
image, and display it. Useful for verifying that config changes
(xsize, ysize, field_center, etc.) take effect.

Run:
    python -m fpwfsc.sim.sim_camera_image_generator
"""

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import fpwfsc.common.support_functions as sf
import fpwfsc.common.fake_hardware as fhw


def main():
    sim_dir = Path(__file__).parent.absolute()
    hw = sf.validate_config(str(sim_dir / 'sim_config.ini'),
                            str(sim_dir / 'sim_config.spec'))

    cam = hw['SIMULATION']['CAMERA_PARAMS']
    opt = hw['SIMULATION']['OPTICAL_PARAMS']

    print(f"N pix focal (hcipy grid): {opt['N pix focal']}")
    print(f"Canvas size: {cam['xsize']} x {cam['ysize']}")
    print(f"Field center: ({cam['field_center_x']}, {cam['field_center_y']})")

    CSM = fhw.FakeCoronagraphOpticalSystem(**opt)
    Camera = fhw.FakeDetector(opticalsystem=CSM, **cam)

    img = Camera.take_image()
    print(f"Output shape: {img.shape}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    vmin, vmax = np.percentile(img, [1, 99])
    ax1.imshow(img, origin='lower', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Full frame {img.shape[1]}x{img.shape[0]}  (1-99% stretch)')
    ax1.axhline(cam['field_center_y'], color='r', ls='--', alpha=0.5)
    ax1.axvline(cam['field_center_x'], color='r', ls='--', alpha=0.5)

    hw = 50
    cx, cy = cam['field_center_x'], cam['field_center_y']
    cutout = img[max(0,cy-hw):cy+hw, max(0,cx-hw):cx+hw]
    vmin2, vmax2 = np.percentile(cutout, [1, 99])
    ax2.imshow(cutout, origin='lower', vmin=vmin2, vmax=vmax2)
    ax2.set_title(f'Cutout around ({cx}, {cy})')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
