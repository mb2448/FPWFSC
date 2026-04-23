# miniQACITS

miniQACITS is a closed-loop tip-tilt correction system for coronagraphic (or non-coronagraphic) imaging. It measures the position of a star on a detector using a quad cell flux method and applies corrections to the AO system to keep the star centered at a particular location.

---

## How it works

Each iteration of the control loop:

1. **Acquires an image** from the camera (or reads the latest one from disk in "hitchhiker mode")
2. **Applies calibrations (optional)** — background subtraction, flat fielding, bad pixel correction
3. **Crops the image** to a square region around the expected star position
4. **Measures the centroid offset** using a quad cell flux calculation over an annular region
5. **Feeds the offset into a PID loop** to compute a correction in centroid units
6. **Applies rotation/flip calibration** to convert from detector frame to AO frame
7. **Sends the correction** to the AO system 

### Quad cell measurement

The quad cell divides an annular region (defined by an inner and outer radius) into four quadrants around the expected star position. The flux in each quadrant (A, B, C, D) is summed, and the centroid offset is:

```
x_offset = (B + D - A - C) / (A + B + C + D)
y_offset = (A + B - C - D) / (A + B + C + D)
```

Offsets are in **array-index convention**: positive x_offset means the star is at a larger column index (right in display); positive y_offset means larger row index (up in display). The result is dimensionless (normalised by total flux), so should be the same all stars (assuming good background subtraction). The PID + tip-tilt calibration (gain, angle, flips) handles the mapping from these offsets to the correct AO correction direction.

### Background subtraction

When a background FITS file is provided, it is subtracted from each frame. When no background file is provided, a 2D background is estimated automatically from the border rows and columns of the full frame (column medians from the outer 5 rows, then row medians from the outer 5 columns of the column-subtracted image). The background is estimated from the **full frame** before cropping, so the PSF does not contaminate the estimate.

---

## Running the GUI

```bash
python qacits_GUI.py
```

### GUI layout (top to bottom)

**Hardware selector** — dropdown at the top to choose `Sim` or `NIRC2`. The camera and AO system are loaded immediately on selection. If NIRC2 fails to load (e.g. not on the Keck machine), it falls back to Sim.

**EXECUTION section:**
- `plot` — enable/disable live image plotting
- `N iterations` — number of loop iterations
- `setpoint: x:[ ] y:[ ]` — inline text fields for the expected star pixel coordinates
- **Apply New Setpoint** (green button) — pushes the setpoint values to the running loop without stopping it. Greyed out until the loop is running.
- Expert options (collapsed): `inner radius`, `outer radius`

**PID section:**
- `centroid offset: x:[ ] y:[ ]` — inline text fields for the quad cell target offset. Default (0, 0) means drive the star to the setpoint pixel. Nonzero values make the PID maintain a deliberate offset.
- **Set Offset** (green) / **Capture Offset** (blue) — two half-width buttons:
  - **Set Offset** pushes the typed centroid offset values to the running PID.
  - **Capture Offset** tells the loop to grab the current quad cell reading and use it as the new PID target. The button latches ("Capturing...") until the loop applies it on the next iteration, then the text fields update with the captured values.
- Expert options (collapsed): `Kp`, `Ki`, `Kd`, `output_limits`

**CAMERA CALIBRATION section** (all expert, collapsed):
- File pickers for background, masterflat, and bad pixel map FITS files. Each has a text field + browse button. **Text fields turn orange** if the specified file does not exist on disk.

**HITCHHIKER MODE section:**
- `hitchhike` toggle
- Expert options (collapsed): `imagedir` (directory picker), `poll interval`, `timeout`

**AO section** (all expert, collapsed):
- `tip tilt gain`, `tip tilt angle (deg)`, `tip tilt flip x`, `tip tilt flip y`

**Bottom buttons:**
- **Save configuration** / **Load configuration** — save or load `.ini` files
- **Run** / **Stop** — starts or stops the control loop. On Stop, the loop finishes its current iteration gracefully (no force-terminate). The Run button re-enables once the thread exits.
- **Reset DTT Offset** (orange) — zeros the AO tip-tilt offsets. Only works with real hardware (prints a message in sim mode).

### Live image plotter

When `plot = True`, a separate window opens showing the cropped image with:
- Inner and outer annulus circles (red). Inner circle can be 0.
- Crosshair lines between the annuli
- Center point marker
- Title showing the current centroid offset (x, y)
- Percentile stretch sliders (Min/Max) at the bottom
- **Pixel value readout** — hover over the image to see `x=... y=... val=...` in a label at the bottom

The plotter **stays open when Stop is pressed** so you can inspect the last frame. It closes when Run is pressed again (a fresh plotter opens) or when the main GUI window is closed.

### Error handling

- If the algorithm thread crashes (e.g. calibration file shape mismatch), a **popup dialog** appears with the error message.
- Calibration file path fields turn **orange** when the file does not exist.
- `load_fits_or_none` prints a **WARNING** to the terminal when a path is specified but the file is missing.
- Calibration file paths are printed to the terminal at loop start for verification.

---

## Running from the command line

```bash
python run_qacits.py
```

This runs in simulation mode using `qacits_config.ini` and opens a live plot window.

---

## Configuration

All settings are stored in `qacits_config.ini` and validated against `qacits_config.spec`. A simulation-specific preset is available in `qacits_config_sim.ini`.

### [EXECUTION]

| Parameter | Description |
|-----------|-------------|
| `plot` | Show the live image window during the loop |
| `N iterations` | Number of loop iterations to run |
| `x setpoint` | Expected X pixel coordinate of the star on the detector |
| `y setpoint` | Expected Y pixel coordinate of the star on the detector |
| `inner radius` | Inner radius of the quad cell annulus in pixels. Set to 0 for a full circle (no inner exclusion). |
| `outer radius` | Outer radius of the quad cell annulus in pixels. Must be > inner radius. Also sets the crop size (2 * outer_radius + 1). |

### [PID]

| Parameter | Description |
|-----------|-------------|
| `x centroid offset` | Quad cell X target in centroid units (0 = centered on setpoint). Range -1 to 1. |
| `y centroid offset` | Quad cell Y target in centroid units (0 = centered on setpoint). Range -1 to 1. |
| `Kp` | Proportional gain. Start with 0.3-0.5. |
| `Ki` | Integral gain. Start at 0; add small values to eliminate steady-state offset. |
| `Kd` | Derivative gain. Usually 0. |
| `output_limits` | Maximum PID output magnitude in centroid units. Symmetric +/- clamp. Prevents large corrections from noisy measurements. Default 0.5. |

The PID controller operates in **centroid units** (dimensionless quad cell flux ratios, roughly -1 to +1). Its output is multiplied by `tip tilt gain` and passed through `rotate_flip_tt` to produce the AO command. The integral term has built-in anti-windup: it is clamped to `output_limits` each iteration.

Setting `Ki = 0` and `Kd = 0` gives a pure proportional controller, which is the recommended starting point.

### [CAMERA CALIBRATION]

| Parameter | Description |
|-----------|-------------|
| `background file` | Path to a FITS background/dark frame. If blank, background is estimated from the border rows/columns of each frame. |
| `masterflat file` | Path to a mean-normalized FITS master flat. If blank, no flat correction is applied. |
| `badpix file` | Path to a FITS bad pixel map (non-zero = bad). If blank, no bad pixel correction is applied. |

All calibration files must match the camera image dimensions exactly; a shape mismatch raises an error popup on the first iteration. Use `fpwfsc/calibration/detector_calibration_GUI.py` to generate these files.

### [HITCHHIKER MODE]

| Parameter | Description |
|-----------|-------------|
| `hitchhike` | If True, the loop reads images from disk instead of controlling the camera directly |
| `imagedir` | Directory to watch for new FITS files |
| `poll interval` | How often (seconds) to check the directory for a new file |
| `timeout` | If no new file appears within this many seconds, the loop exits with an error |

Hitchhiker mode is useful when another process owns the camera. The loop waits for a new FITS file to appear in `imagedir`, processes it, then waits for the next one.

### [AO]

| Parameter | Description |
|-----------|-------------|
| `tip tilt gain` | Conversion factor from centroid units to AO tip-tilt command units. Sign and magnitude are instrument-dependent. |
| `tip tilt angle (deg)` | Rotation angle between detector axes and AO axes. Calibrate per instrument. |
| `tip tilt flip x` | Flip (negate) the X correction direction |
| `tip tilt flip y` | Flip (negate) the Y correction direction |

The gain, angle, and flips are applied via `dm.rotate_flip_tt` before calling `AOSystem.offset_tiptilt(x, y)`. This calibration maps from detector-frame centroid corrections to AO-native commands.

---

## Updating the setpoint mid-loop

The pixel setpoint can be changed while the loop is running. Edit the `x` and `y` fields in the EXECUTION section and click **Apply New Setpoint**. The new setpoint takes effect on the next iteration. The PID controller is not reset — its integral state is preserved.

## Centroid offset (holding a deliberate offset)

By default the PID drives the quad cell reading to (0, 0) — star centered on the setpoint pixel. To hold the star at a deliberate offset:

- **Type values** into the centroid offset fields and click **Set Offset**, or
- **Click Capture Offset** to grab the current quad cell reading as the new target. The button latches ("Capturing...") until the loop applies it on the next iteration, then the text fields update with the captured values.

**Pre-arming Capture Offset:** Capture Offset can be clicked **before starting the loop**. This is useful when someone else has already centered the coronagraph and you want the loop to maintain that exact position. Click Capture Offset (it latches), then click Run — the first iteration will measure the current centroid and use it as the PID target. The star stays exactly where it was.

The centroid offset is in dimensionless quad cell units (range -1 to 1). The PID maintains whatever offset is set, so the star stays at that position relative to the setpoint pixel.

Both mechanisms are implemented via thread-safe queues: the GUI puts the new offset into `centroid_offset_queue`, and the control loop reads from it at the start of each iteration.

---

## Simulation mode

When hardware is set to `Sim`, the code uses a simulated optical system, DM, and camera defined in `fpwfsc/sim/sim_config.ini`. The simulator generates realistic PSF images including a coronagraphic vortex effect, read noise, dark current, and deterministic bad pixels (controlled by `bad_pixel_fraction` in the sim config). An initial tip-tilt offset is applied at startup so the loop has a non-zero error to correct.

The sim config also includes a `readout_delay` parameter (seconds) to slow down the loop for interactive testing. Set to 0 for full speed.

### Hitchhiker mode with simulation

To test hitchhiker mode with the simulator, use the sim image generator in a separate terminal:

```bash
# Stream FITS files to a directory
python -m fpwfsc.sim.sim_image_generator /path/to/output/directory --interval 1.0

# Or show a single diagnostic image
python -m fpwfsc.sim.sim_image_generator --show
```

Then configure `imagedir` in the GUI to the same directory and enable `hitchhike`. The control loop will process each file as it appears.

Options for the image generator:

```
--interval FLOAT    Time between images in seconds (default: 1.0)
--count INT         Number of images to generate (default: infinite)
--no-drift          Disable random tip-tilt drift between frames
--show              Show a single image and exit (diagnostic mode)
```

### NIRC2 hitchhiker testing

To test hitchhiker mode with the real NIRC2 camera (on Keck):

```bash
python -i fpwfsc/qacits/nirc2_image_grabber.py
>>> save_image('/path/to/imagedir')
```

Each call grabs a fresh NIRC2 frame and writes it as a FITS file.

---

## Detector calibration

Use the calibration GUI to take backgrounds, flats, and bad pixel maps:

```bash
python -m fpwfsc.calibration.detector_calibration_GUI
```

This tool works with both Sim and NIRC2. Output files:
- `masterbgd.fits` — median of N frames
- `masterflat.fits` — mean-normalized median of N frames
- `badpix.fits` — sigma-clipped bad pixel map (Gaussian fit to pixel intensity histogram)

Point the qacits CAMERA CALIBRATION fields at these files.

---

## AO interface

Both sim and real AO expose the same interface:

```python
aosystem.offset_tiptilt(x, y)    # add (x, y) to current tip/tilt
aosystem.zero_tiptilt()          # reset tip/tilt to zero
```

- **Sim:** `FakeAODMSystem` synthesises a DM tilt surface via `dm.generate_tip_tilt`.
- **NIRC2 / Keck:** `K2AOAlias` (in `qacits_hardware.py`) wraps the K2AO FSM via `aoscripts.ao_systems.k2ao`.

Rotation/flip is applied **before** calling `offset_tiptilt`, not inside it. This keeps the interface clean for FSM targets that have no DM shape.

---

## File overview

| File | Purpose |
|------|---------|
| `qacits_GUI.py` | Main PyQt5 GUI — config editing, run/stop, live plot, setpoint/offset controls |
| `run_qacits.py` | Control loop — acquires images, computes corrections, drives the AO |
| `qacits_funcs.py` | Core math — image cropping and quad cell flux computation |
| `PID.py` | PID controller with numpy array support for 2D (x, y) control |
| `qacits_plotter_qt.py` | Live image display window with annulus overlay, stretch controls, pixel readout |
| `qacits_gui_helper.py` | GUI metadata — instrument list, config field descriptions, expert flags |
| `qacits_hardware.py` | Real hardware wrappers — `NIRC2Alias` (camera) and `K2AOAlias` (FSM) |
| `nirc2_image_grabber.py` | Interactive tool to grab NIRC2 frames for hitchhiker testing |
| `qacits_config.ini` | User configuration file (may contain Keck-specific paths) |
| `qacits_config_sim.ini` | Simulation-specific configuration preset |
| `qacits_config.spec` | Configuration schema and default values |

Simulation tools live in `fpwfsc/sim/`:
| File | Purpose |
|------|---------|
| `sim_config.ini` / `sim_config.spec` | Simulated optical system, camera, and AO parameters |
| `sim_image_generator.py` | Generate sim images for display (`--show`) or hitchhiker streaming (`<dir>`) |

Calibration tools live in `fpwfsc/calibration/`:
| File | Purpose |
|------|---------|
| `detector_calibration_GUI.py` | PyQt5 tool to take BGD / FLAT / BADPIX |
| `detector_calibration_script.py` | CLI calibration walkthrough |

---

## Tuning guide

**Starting out:**
1. Set `Ki = 0`, `Kd = 0`
2. Set `Kp` to a small value (e.g. 0.3-0.5)
3. Check that corrections move the star in the right direction; if not, adjust `tip tilt angle (deg)` or the flip flags
4. Increase `Kp` until the loop converges quickly without oscillating

**Annulus sizing:**
- The inner radius should exclude the PSF core (or coronagraph inner working angle) where flux varies nonlinearly with offset. Set to 0 to include the full circle.
- The outer radius should be large enough to capture sufficient flux but small enough to avoid contamination from other sources. Also determines the crop size.
- A typical starting point: inner = 10 px, outer = 30 px

**Output limits:**
- `output_limits` prevents large corrections from being applied if the star is lost or the measurement is noisy
- The value is in centroid units (quad cell flux ratio range is roughly -1 to +1)
- Default 0.5 is a reasonable starting point

**Centroid offset:**
- Use **Capture Offset** to lock the current star position as the PID target
- Pre-arm it before Run to maintain an existing alignment from the start
- Use **Set Offset** to type a specific offset value
- Set both to 0 to return to normal centering behavior
