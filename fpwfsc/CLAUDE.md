# FPWFSC — Codebase guide for Claude

Python package for focal-plane wavefront sensing and control at W.M. Keck
Observatory (NIRC2, OSIRIS, SCALES) and Subaru/SCExAO (VAMPIRES, Palila).
Physics engine is `hcipy`.

## Package structure

```
fpwfsc/
├── common/              Shared infrastructure used by every pipeline.
│   ├── support_functions.py  image reduction, config validation, utilities
│   ├── fake_hardware.py      FakeCoronagraphOpticalSystem / FakeDetector /
│   │                         FakeAODMSystem / FakeAOSystem / Hitchhiker
│   ├── bench_hardware.py     real-hardware wrappers (not usable off-instrument)
│   ├── dm.py                 DM/FSM geometry helpers (generate_tip_tilt,
│   │                         rotate_flip_tt, waffle, speckles)
│   ├── classes.py            FastandFurious algorithm + optical system model
│   ├── plotting_funcs.py     Matplotlib helpers
│   └── LS{1,2,3}percent.npy  Lyot-stop masks (SCALES)
├── sim/                 Simulation config + image generation tools.
│   ├── sim_config.ini        Canonical sim params (optical, camera, AO)
│   ├── sim_config.spec       Matching spec
│   └── sim_image_generator.py  --show (diagnostic) or <dir> (FITS stream)
├── calibration/         Detector calibration tools (shared across pipelines).
│   ├── detector_calibration_GUI.py   PyQt5 BGD/FLAT/BADPIX tool
│   └── detector_calibration_script.py  CLI calibration walkthrough
├── fnf/                 Fast-and-Furious WFS algorithm.
├── qacits/              Coronagraph tip-tilt centering (quad-cell + PID).
├── san/                 Speckle Nulling algorithm.
├── satellite_pointing_control/  Satellite spot pointing loop.
├── bgds/                Shipped calibration FITS (medbackground.fits, etc.).
└── speckle_nulling_old_code/    Legacy archive. Do not import.
```

Each pipeline owns its own `run.py` / `run_xxx.py` + `*_GUI.py` + config ini/spec.

## Architecture pattern

All pipelines follow the same three-layer pattern:

1. **GUI** — presents config fields, collects user input, writes to a validated
   config object. Knows about widgets, not about physics.
2. **Config (.ini + .spec)** — the complete, portable contract between the GUI
   and the algorithm. Editable by hand, shareable, version-controllable. The
   `.spec` enforces types/ranges so invalid configs fail at load time.
3. **Run script** — reads config, executes the control loop, does not know or
   care whether a GUI or command line produced the config. Plotter is an
   optional side channel, injected via signal.

This means every run is fully described by its `.ini` file, the algorithm can
be tested without the GUI, and multiple entry points (GUI, CLI, Jupyter) can
all call the same `run()`.

## Core conventions

### Sim vs real
Every `run()` has a branch:

```python
if camera == 'Sim' and aosystem == 'Sim':
    # instantiate FakeCoronagraphOpticalSystem, FakeDetector, FakeAODMSystem
    # from sim/sim_config.ini
else:
    Camera = camera
    AOSystem = aosystem
```

Pass string sentinels `'Sim', 'Sim'` for sim; pass real-hardware-wrapper instances
otherwise. The GUIs route to this via `load_instruments(instrumentname)` in their
respective `gui_helper.py`.

Non-san pipelines (qacits, satellite, calibration GUI) read sim params from
`fpwfsc/sim/sim_config.ini`. San scripts still read from their own
`san/sn_config.ini` (which retains a `[SIMULATION]` section for backward compat).

### Config files
`.ini` with matching `.spec`, validated through
`sf.validate_config(path_to_ini, path_to_spec)` which builds a ConfigObj and
runs `MyValidator` (adds `*_or_none` types on top of stock validate).

### Threading (GUIs)
- `QThread` runs the control loop / acquisition.
- `threading.Event` signals stop.
- `queue.Queue` pushes runtime parameter updates into the loop (e.g.,
  mid-loop setpoint changes in qacits).
- All plotting must go through a `pyqtSignal` connected back to the main
  thread — macOS requires GUI ops on the main thread.

### Hitchhiker mode
`fhw.Hitchhiker(imagedir, poll_interval, timeout)` watches a directory for new
FITS files written by an external process and returns them to a loop in place
of a direct `Camera.take_image()`. Each pipeline's config has its own
`hitchhike` toggle. Test with `sim/sim_image_generator.py <dir>` or
`qacits/nirc2_image_grabber.py` (real camera).

### Calibration
`sf.equalize_image(data, bkgd=None, masterflat=None, badpix=None)` handles
background subtraction, flat division, bad-pixel replacement. Missing inputs
fall back to sensible defaults: `bkgd` → 2D border-median estimation via
`estimate_background_from_border` (column median then row median from the
outer 5 rows/cols), `masterflat` → 1, `badpix` → unchanged.
`sf.load_fits_or_none(path)` loads a FITS if the path is non-empty and exists,
else returns None.

Bad-pixel replacement uses the local 5×5 median around each flagged pixel
(`sf.removebadpix`, scales with bad-pixel count, not image size).

Bad-pixel mask generation: `sf.locate_badpix(data, sigmaclip=3, plot=True)`
fits a Gaussian to the pixel-intensity histogram and sigma-clips the tails.
Plot uses log-y, symlog-x.

### Simulated bad pixels
`fhw.deterministic_bad_pixels(height, width, fraction)` generates a
reproducible bad-pixel mask from pixel coordinates using a Murmur3-style
bit-mixing hash. Same pixels flagged regardless of image size; no stored
file needed. Controlled via `bad_pixel_fraction` in `sim_config.ini`.

## AO interface (post Option A refactor)

Both sim and real AO expose the same two methods:

```python
aosystem.offset_tiptilt(x, y)    # add (x, y) to current tip/tilt command
aosystem.zero_tiptilt()          # reset tip/tilt
```

- Sim side: `FakeAODMSystem` implements these by synthesizing a DM surface
  via `dm.generate_tip_tilt` and updating the shape.
- Real side (Keck): `qacits/qacits_hardware.K2AOAlias` wraps the K2AO FSM.

Rotation/flip is **not** part of the AO method signature. Callers must apply
`dm.rotate_flip_tt(x, y, rot_deg, flipx, flipy)` **before** calling
`offset_tiptilt`. This keeps FSM-only systems (no DM shape) happy.

`FakeAOSystem` (the older DM-less class) still exists; fnf uses it. It does
not have `offset_tiptilt`.

## Detector calibration tools

Two entry points:

- `calibration/detector_calibration_script.py` — text-prompt CLI that walks
  through bgd / flat / flatdark and synthesizes badpix + masterflat.
- `calibration/detector_calibration_GUI.py` — PyQt5 button-driven tool with
  three buttons (BGD / FLAT / BADPIX), confirmation dialog, per-frame progress
  reporting, matplotlib badpix plot, recommendations table.

Output filenames:
- `masterbgd.fits` — median of N frames (GUI only).
- `masterflat.fits` — mean-normalized median (GUI) or
  `removebadpix(flat - flatdark) / mean` (CLI).
- `badpix.fits` — sigma-clipped bad-pixel map via `locate_badpix`.

## Important gotchas

### Installation
Use `pip install -e .` from the repo root (has `pyproject.toml`). Running
scripts directly from their directory without installing the package
produces `ModuleNotFoundError: No module named 'fpwfsc'`. After pulling new
code that adds subpackages (e.g., `sim/`, `calibration/`), re-run
`pip install -e .` to register them.

### FakeDetector save-to-disk
`FakeDetector.take_image(output_directory=...)` saves sim frames to disk if
a directory is passed. Most callers leave it blank;
`sim/sim_image_generator.py` uses it intentionally for hitchhiker testing.
A one-line "Sim image acquired at ..." prints on every sim frame regardless.

### matplotlib + Qt on macOS
Any Qt GUI that ends up calling `plt.show()` must force the Qt5Agg backend
before any pyplot import:
```python
import matplotlib
matplotlib.use('Qt5Agg')
```
Otherwise you get `ImportError: Cannot load backend 'MacOSX' ... as 'qt' is
currently running`. See top of `calibration/detector_calibration_GUI.py`.

### center_image coordinate convention
`center_image(small, large_size, center_position)` in `fake_hardware.py`
takes `center_position = (x, y)` where x = column, y = row. Internally
indexes as `large_image[row:, col:]`. Clips gracefully if the small image
extends past the canvas boundary. Historically had a coordinate swap bug
that placed PSFs at transposed positions — fixed Apr 2026.

### `locate_badpix` overflow
`100 * np.sum(bpmask)` overflowed when bpmask was a small-int dtype. Fixed
by casting to float before multiplying.

### FakeAODMSystem `dm_rotation_angle` parameter
`FakeAODMSystem.__init__` accepts `rotation_angle_dm` but warns it is not
implemented in the simulator. Don't rely on sim honoring clocking.

### `sn_config.ini` hardcoded paths
`bgddir` under `CAMERA_CALIBRATION` may still contain machine-specific paths.
San scripts that call `sf.setup_bgd_dict(bgddir)` will fail on other machines
until that path is edited.

## qacits specifics

- `run_qacits.py` runs the centroiding + PID + tip-tilt loop.
- Quad cell: `qacits_funcs.compute_quad_cell_flux(image, x_center, y_center,
  min_radius, max_radius)` returns fractional `(x_offset, y_offset)` from
  annulus flux sums.
- PID: `qacits/PID.py`, numpy-array aware, derivative-on-measurement,
  anti-windup via symmetric output_limits clamping of the integral.
- GUI: `qacits_GUI.py`. Hardware dropdown (Sim/NIRC2), editable config with
  expert-options collapsibles, mid-loop setpoint update via `queue.Queue`.
  Plotter stays open on Stop (shows last frame for inspection).
- Plotter: `qacits_plotter_qt.py` (pyqtgraph). Shows cropped image with
  annulus overlay, percentile stretch sliders, pixel-value readout on hover.
- Hitchhiker: enable `hitchhike = True` in config; loop reads from `imagedir`
  each iteration instead of calling `take_image`.
- `qacits_hardware.py`: `NIRC2Alias` + `K2AOAlias`. `NIRC2Alias.take_image()`
  returns the raw data array (not the HDU). `K2AOAlias` interacts with the
  real Keck FSM via `aoscripts.ao_systems.k2ao.K2AO()` — only importable on
  the Keck box.

## fnf specifics

- `run.py` runs the Fast-and-Furious phase-estimation + DM-correction loop.
- Algorithm lives in `common/classes.py` (`FastandFurious` class). Uses an
  internal forward model (`ff_c.SystemModel`) for phase estimation.
- Plotter: `FF_plotter_qt.py` (pyqtgraph). 2x2 grid: PSF, wavefront
  residuals, Strehl ratio, contrast curve. Pixel-value readout on hover for
  the two image panels.
- In sim mode, `FakeDetector` and `FakeAOSystem` share the same `OpticalModel`
  that FnF uses as its forward model. This is an "inverse crime" — the
  algorithm solves a problem whose answer it already knows. See future plans.
- Camera params are currently hardcoded in `run.py` sim branch (lines 108-118)
  instead of loaded from `sim_config.ini`. See future plans.

## Typical workflows

### Running qacits in sim mode locally
```
pip install -e .
python fpwfsc/qacits/qacits_GUI.py
# Pick 'Sim', click Run
```

### Running qacits on Keck with NIRC2 (no AO)
```
python fpwfsc/qacits/qacits_GUI.py
# Pick 'NIRC2', enable hitchhike, point imagedir at a test directory
# In another terminal:
python -i fpwfsc/qacits/nirc2_image_grabber.py
>>> save_image('/path/to/imagedir')
```

### Sim image generation for hitchhiker testing
```
# Show a single image (diagnostic)
python -m fpwfsc.sim.sim_image_generator --show

# Stream FITS to a directory
python -m fpwfsc.sim.sim_image_generator /path/to/dir --interval 1.0
```

### Taking detector calibrations
```
python -m fpwfsc.calibration.detector_calibration_GUI
# Pick instrument, set output dir, click Take BGD / FLAT / BADPIX
```

### Committing and pushing
Github repo: `mb2448/FPWFSC`, branch `main`. `pip install -e .` survives pulls.
When working with multiple developers, push to feature branches or coordinate
to avoid divergence; remote-history rebase is the usual reconciliation.

## Future plans

### Decouple fnf sim from algorithm forward model (planned, not started)

Currently fnf sim mode shares a single `OpticalModel` between the FnF
algorithm (as its forward model) and the fake hardware (for image generation).
This creates an inverse crime — the algorithm is always solving with a
perfect model, so sim results are unrealistically optimistic.

**Target architecture:** two separate optical models:
- `SimModel` (from `sim/sim_config.ini`) — the "truth" that generates images.
  Uses `FakeCoronagraphOpticalSystem` + `FakeAODMSystem` + `FakeDetector`.
- `AlgoModel` (from `FF_software.ini [MODELLING]`) — FnF's internal forward
  model, which may have intentional mismatches (wrong rotation, pixel scale
  error, no knowledge of non-common-path aberrations, etc.).

**Key change:** switch fnf sim from `FakeAOSystem` (mode-based) to
`FakeAODMSystem` (actuator-based) so the DM command interface is independent
of the algorithm's mode basis. FnF already produces 2D microns arrays, so
the interface fits.

**Scope:** ~50 lines in `fnf/run.py` sim branch. No algorithm changes, no
changes to FakeHardware classes. Optional `[SIM_MISMATCH]` config section
to dial in specific model errors (rotation offset, pixel scale error, extra
WFE, center offset).

**Why it matters:** when FnF "works in sim but not on-sky," the first thing
to check is model mismatch sensitivity. This architecture enables that test
without leaving the simulator.

### Also load fnf camera params from sim_config.ini

fnf sim currently hardcodes camera params (xsize=1024, ysize=1024,
field_center_x=330, etc.) in `run.py` lines 108-118 instead of reading from
`sim_config.ini`. Should be changed to load from config like qacits does,
so changing `sim_config.ini` affects all pipelines consistently.

### Add CAMERA CALIBRATION section to satellite config

`satellite_pointing_control/satellite_config.ini` has no calibration fields.
Currently bgds default to all-None. When real calibration files are needed
for satellite pointing, add a `[CAMERA CALIBRATION]` section matching the
qacits/fnf pattern and wire it through `load_fits_or_none`.
