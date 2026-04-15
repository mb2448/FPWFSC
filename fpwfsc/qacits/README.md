# miniQACITS

miniQACITS is a closed-loop tip-tilt correction system for coronagraphic imaging. It measures the position of a star on a detector using a quad cell flux method and applies corrections to the AO deformable mirror to keep the star centred on the coronagraph.

---

## How it works

Each iteration of the control loop:

1. **Acquires an image** from the camera (or reads one from disk in hitchhiker mode)
2. **Applies calibrations** — background subtraction, flat fielding, bad pixel correction
3. **Crops the image** to a small region around the expected star position
4. **Measures the tip-tilt error** using a quad cell flux calculation over an annular region
5. **Feeds the error into a PID controller** to compute a correction
6. **Applies the correction** to the DM as a tip-tilt mode

### Quad cell measurement

The quad cell divides an annular region (defined by an inner and outer radius) into four quadrants around the expected star position. The flux in each quadrant (A = top-left, B = top-right, C = bottom-left, D = bottom-right) is summed, and the centroid offset is:

```
x_offset = (B + D - A - C) / (A + B + C + D)
y_offset = (A + B - C - D) / (A + B + C + D)
```

A positive x_offset means the star is to the right of centre; a positive y_offset means it is above centre. The result is dimensionless (normalised by total flux) and represents a fractional displacement within the annulus.

---

## Running the GUI

```bash
python qacits_GUI.py
```

The GUI provides:

- **Hardware selection** — choose the instrument or simulation mode
- **Configuration editing** — all parameters editable in-window; expert options collapsed by default
- **Save / Load configuration** — save or load `.ini` files
- **Run / Stop** — starts or stops the control loop
- **Update Setpoint** — sends a new setpoint to the running loop without resetting the PID

### Expert options

Parameters marked as expert are hidden by default and accessible via the collapsible "Expert Options" section within each config group. These include PID gains, tip-tilt calibration, annulus radii, and calibration file paths.

---

## Running from the command line

```bash
python run_qacits.py
```

This runs in simulation mode using `qacits_config.ini` and opens a live plot window.

---

## Configuration

All settings are stored in `qacits_config.ini` and validated against `qacits_config.spec`.

### [EXECUTION]

| Parameter | Description |
|-----------|-------------|
| `plot` | Show the live image window during the loop |
| `N iterations` | Number of loop iterations to run |
| `x setpoint` | Expected X pixel coordinate of the star |
| `y setpoint` | Expected Y pixel coordinate of the star |
| `inner radius` | Inner radius of the quad cell annulus (pixels) |
| `outer radius` | Outer radius of the quad cell annulus (pixels). Must be strictly greater than inner radius. |

The setpoint is the pixel coordinate where the star is expected to be. The quad cell measures offsets relative to this point.

### [CAMERA CALIBRATION]

| Parameter | Description |
|-----------|-------------|
| `background file` | Path to a FITS background/dark frame. If blank, the image median is subtracted instead. |
| `masterflat file` | Path to a FITS master flat. If blank, no flat correction is applied. |
| `badpix file` | Path to a FITS bad pixel map (non-zero = bad). If blank, no bad pixel correction is applied. |

All calibration files must match the camera image dimensions exactly; a mismatch raises an error on the first iteration.

### [HITCHHIKER MODE]

| Parameter | Description |
|-----------|-------------|
| `hitchhike` | If True, the loop reads images from disk instead of controlling the camera directly |
| `imagedir` | Directory to watch for new FITS files |
| `poll interval` | How often (seconds) to check the directory for a new file |
| `timeout` | If no new file appears within this many seconds, the loop exits with an error |

Hitchhiker mode is useful when another process owns the camera. The loop waits for a new FITS file to appear in `imagedir`, processes it, then waits for the next one.

### [PID]

| Parameter | Description |
|-----------|-------------|
| `Kp` | Proportional gain |
| `Ki` | Integral gain |
| `Kd` | Derivative gain |
| `output_limits` | Maximum PID output magnitude in pixels. Acts as a symmetric ±limit on the correction applied each iteration. |

The PID controller operates in pixel units. Its output is multiplied by `tip tilt gain` to produce the DM command. The integral term has built-in anti-windup: it is clamped to `output_limits` each iteration.

Setting `Ki = 0` and `Kd = 0` gives a pure proportional controller, which is the recommended starting point.

### [AO]

| Parameter | Description |
|-----------|-------------|
| `tip tilt gain` | Conversion factor from pixels to DM tip-tilt units |
| `tip tilt angle (deg)` | Rotation angle between detector axes and DM axes |
| `tip tilt flip x` | Flip the X correction direction |
| `tip tilt flip y` | Flip the Y correction direction |

The tip-tilt gain, angle, and flips must be calibrated for each instrument to ensure the DM correction moves the star in the correct direction on the detector.

---

## Updating the setpoint mid-loop

The setpoint can be changed while the loop is running without stopping it. In the GUI, edit the `x setpoint` and `y setpoint` fields and click **Update Setpoint**. The new setpoint takes effect on the next iteration. The PID controller is not reset — its integral state is preserved.

This is implemented via a thread-safe `queue.Queue`: the GUI puts the new setpoint into the queue, and the control loop reads from it at the start of each iteration.

---

## Simulation mode

When hardware is set to `Sim`, the code uses a simulated optical system, DM, and camera defined in `../san/sn_config.ini`. The simulator generates realistic PSF images including a coronagraphic effect. An initial tip-tilt offset is applied at startup so the loop has a non-zero error to correct.

### Hitchhiker mode with simulation

To test hitchhiker mode with the simulator, run the image generator in a separate terminal:

```bash
python qacits_camera_image_generator.py /path/to/output/directory --interval 1.0
```

Then configure `imagedir` in the GUI to the same directory and enable `hitchhike`. The control loop will process each file as it appears.

Options for the image generator:

```
--interval FLOAT    Time between images in seconds (default: 1.0)
--count INT         Number of images to generate (default: infinite)
```

---

## File overview

| File | Purpose |
|------|---------|
| `qacits_GUI.py` | Main PyQt5 GUI — config editing, run/stop, live plot |
| `run_qacits.py` | Control loop — acquires images, computes corrections, drives the DM |
| `qacits_funcs.py` | Core math — image cropping and quad cell flux computation |
| `PID.py` | PID controller with numpy array support for 2D (x, y) control |
| `qacits_plotter_qt.py` | Live image display window with annulus overlay and stretch controls |
| `qacits_gui_helper.py` | GUI metadata — instrument list, config field descriptions, expert flags |
| `qacits_camera_image_generator.py` | Standalone tool to generate simulated FITS images for hitchhiker testing |
| `qacits_config.ini` | User configuration file |
| `qacits_config.spec` | Configuration schema and default values |

---

## Tuning guide

**Starting out:**
1. Set `Ki = 0`, `Kd = 0`
2. Set `Kp` to a small value (e.g. 0.3–0.5)
3. Check that corrections move the star in the right direction; if not, adjust `tip tilt angle (deg)` or the flip flags
4. Increase `Kp` until the loop converges quickly without oscillating

**Annulus sizing:**
- The inner radius should exclude the PSF core (or coronagraph inner working angle) where flux varies nonlinearly with offset
- The outer radius should be large enough to capture sufficient flux but small enough to avoid contamination from other sources
- A typical starting point: inner = 10 px, outer = 30 px

**Output limits:**
- `output_limits` prevents large corrections from being applied if the star is lost or the measurement is noisy
- Set it to the maximum correction (in pixels) that is safe to apply in a single iteration
