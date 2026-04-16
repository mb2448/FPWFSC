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
├── fnf/                 Fast-and-Furious WFS algorithm.
├── qacits/              Coronagraph tip-tilt centering (quad-cell + PID).
├── san/                 Speckle Nulling algorithm + detector calibration tools.
├── satellite_pointing_control/  Satellite spot pointing loop.
├── bgds/                Shipped calibration FITS (medbackground.fits, etc.).
└── speckle_nulling_old_code/    Legacy archive. Do not import.
```

Each pipeline owns its own `run.py` / `run_xxx.py` + `*_GUI.py` + config ini/spec.

## Core conventions

### Sim vs real
Every `run()` has a branch:

```python
if camera == 'Sim' and aosystem == 'Sim':
    # instantiate FakeCoronagraphOpticalSystem, FakeDetector, FakeAODMSystem
    # from the SIMULATION section of sn_config.ini
else:
    Camera = camera
    AOSystem = aosystem
```

Pass string sentinels `'Sim', 'Sim'` for sim; pass real-hardware-wrapper instances
otherwise. The GUIs route to this via `load_instruments(instrumentname)` in their
respective `gui_helper.py`.

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
`hitchhike` toggle.

### Calibration
`sf.equalize_image(data, bkgd=None, masterflat=None, badpix=None)` handles
background subtraction, flat division, bad-pixel replacement. Missing inputs
fall back to sensible defaults: `bkgd` → `np.median(data)` (scalar),
`masterflat` → 1, `badpix` → unchanged. `sf.load_fits_or_none(path)` loads a
FITS if the path is non-empty and exists, else returns None.

Bad-pixel replacement uses the local 5×5 median around each flagged pixel
(`sf.removebadpix`, rewritten Apr 2026 to scale with bad-pixel count, not
image size).

Bad-pixel mask generation: `sf.locate_badpix(data, sigmaclip=3, plot=True)`
fits a Gaussian to the pixel-intensity histogram and sigma-clips the tails.
Pops up a matplotlib histogram figure (log-y, symlog-x).

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

- `san/detector_calibration_script.py` — text-prompt CLI that walks through
  bgd / flat / flatdark and synthesizes badpix + masterflat.
- `san/detector_calibration_GUI.py` — PyQt5 button-driven tool with three
  buttons (BGD / FLAT / BADPIX), confirmation dialog, progress reporting,
  matplotlib badpix plot.

Output filenames:
- `masterbgd.fits` — median of N frames (GUI only).
- `masterflat.fits` — mean-normalized median (GUI) or
  `removebadpix(flat - flatdark) / mean` (CLI).
- `badpix.fits` — sigma-clipped bad-pixel map via `locate_badpix`.

## Important gotchas

### Installation
Use `pip install -e .` from the repo root (has `pyproject.toml`). Running
scripts directly from their directory without installing the package
produces `ModuleNotFoundError: No module named 'fpwfsc'`.

### `sn_config.ini` used to contain hardcoded `/Users/mbottom/...` paths
Fixed Apr 2026 by blanking `output_directory` under `SIMULATION.CAMERA_PARAMS`.
`bgddir` under `CAMERA_CALIBRATION` still points at `/Users/mbottom/...` —
callers that use it (`san/register_dm.py`, `satellite_pointing_control/...`
and others) will fail on non-Mac hosts until the user edits that.

### FakeDetector save-to-disk
`FakeDetector.take_image(output_directory=...)` saves sim frames to disk if
a directory is configured. Most callers leave it blank; `qacits_camera_image_generator.py`
uses it intentionally. The one-line `"Sim image acquired at ..."` print
happens on every sim frame regardless.

### matplotlib + Qt on macOS
Any Qt GUI that ends up calling `plt.show()` must force the Qt5Agg backend
before any pyplot import:
```python
import matplotlib
matplotlib.use('Qt5Agg')
```
Otherwise you get `ImportError: Cannot load backend 'MacOSX' ... as 'qt' is
currently running`. See top of `detector_calibration_GUI.py` for an example.

### qacits `bkgd` scalar fallback
If `background file` is blank, `equalize_image` subtracts `np.median(data)` —
a scalar, not a dark frame. Good enough for ratio-based quad-cell centroiding
in sim, not ideal on-sky. Take a real dark via the calibration GUI.

### `locate_badpix` overflow
Historically `100 * np.sum(bpmask)` overflowed when `bpmask` was a small-int
dtype, producing a `nan%` badge on the histogram plot. Fixed by casting the
sum to `float` before multiplying.

### FakeAODMSystem `dm_rotation_angle` parameter
`FakeAODMSystem.__init__` accepts `rotation_angle_dm` but the docstring warns
it is not implemented in the simulator. Don't rely on sim honoring clocking.

## qacits specifics (the most-edited pipeline)

- `run_qacits.py` runs the centroiding + PID + tip-tilt loop.
- Quad cell: `qacits_funcs.compute_quad_cell_flux(image, x_center, y_center,
  min_radius, max_radius)` returns fractional `(x_offset, y_offset)` from
  annulus flux sums.
- PID: `qacits/PID.py`, numpy-array aware, derivative-on-measurement,
  anti-windup via symmetric output_limits clamping of the integral.
- GUI: `qacits_GUI.py`. Hardware dropdown (Sim/NIRC2), editable config with
  expert-options collapsibles, mid-loop setpoint update via `queue.Queue`.
- Hitchhiker: enable `hitchhike = True` in config; loop reads from `imagedir`
  each iteration instead of calling `take_image`.
- `qacits_hardware.py`: `NIRC2Alias` + `K2AOAlias`. `NIRC2Alias.take_image()`
  returns the raw data array (not the HDU). `K2AOAlias` interacts with the
  real Keck FSM via `aoscripts.ao_systems.k2ao.K2AO()` — only importable on
  the Keck box.

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

### Taking detector calibrations
```
python -m fpwfsc.san.detector_calibration_GUI
# Pick instrument, set output dir, click Take BGD / FLAT / BADPIX
```

### Committing and pushing
Github repo: `mb2448/FPWFSC`, branch `main`. `pip install -e .` survives pulls.
When working with multiple developers, push to feature branches or coordinate
to avoid divergence; remote-history rebase is the usual reconciliation.
