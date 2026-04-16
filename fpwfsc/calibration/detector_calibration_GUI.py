"""
Detector calibration GUI.

Small PyQt5 tool to take backgrounds, flats, and bad-pixel maps with
either the simulator or NIRC2. Mirrors the logic of
calibration/detector_calibration_script.py but wraps it in a simpler,
button-driven interface.

Run:
    python -m fpwfsc.calibration.detector_calibration_GUI
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Qt5Agg')  # must run before any pyplot import so locate_badpix's plot coexists with the Qt GUI

import numpy as np
import astropy.io.fits as fits

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QComboBox, QPushButton,
                             QFileDialog, QMessageBox, QSpinBox,
                             QDialog, QDialogButtonBox)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont

import fpwfsc.common.support_functions as sf


class AcquisitionThread(QThread):
    """Takes N images in a background thread so the UI stays responsive."""
    finished_ok = pyqtSignal(object)  # emits the median image
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)   # (acquired, total)

    def __init__(self, camera, n_images):
        super().__init__()
        self.camera = camera
        self.n_images = n_images

    def run(self):
        try:
            first = self.camera.take_image()
            cube = np.zeros((self.n_images,) + first.shape, dtype=first.dtype)
            cube[0] = first
            self.progress.emit(1, self.n_images)
            for i in range(1, self.n_images):
                cube[i] = self.camera.take_image()
                self.progress.emit(i + 1, self.n_images)
            self.finished_ok.emit(np.median(cube, axis=0))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class ConfirmDialog(QDialog):
    def __init__(self, calibration_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Is the camera set up to take {calibration_name}?"))
        buttons = QDialogButtonBox()
        buttons.addButton("Proceed", QDialogButtonBox.AcceptRole)
        buttons.addButton("Cancel", QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class CalibrationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.camera = None
        self.thread = None
        self._current_kind = None
        self._current_filename = None
        self._init_ui()
        self._load_camera(self.instrument.currentText())

    def _init_ui(self):
        self.setWindowTitle("Detector calibration")
        self.resize(460, 320)
        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Instrument"))
        self.instrument = QComboBox()
        self.instrument.addItems(["Sim", "NIRC2"])
        self.instrument.currentTextChanged.connect(self._load_camera)
        row.addWidget(self.instrument)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Output dir"))
        self.outdir = QLineEdit(str(Path.cwd()))
        row.addWidget(self.outdir)
        browse = QPushButton("...")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse_dir)
        row.addWidget(browse)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("# images"))
        self.n_images = QSpinBox()
        self.n_images.setRange(1, 100)
        self.n_images.setValue(5)
        row.addWidget(self.n_images)
        row.addStretch()
        layout.addLayout(row)

        recommendations = (
            "Recommended setup per calibration type:\n"
            "              default    bench           sky\n"
            "background    all zero   lamp off        slew off star\n"
            "flat          all ones   ignore          sky/dome flat or ignore\n"
            "badpixel      all zero   shutter close   shutter close"
        )
        table_label = QLabel(recommendations)
        table_label.setFont(QFont("Courier", 11))
        layout.addWidget(table_label)

        self.status = QLabel("Ready")
        layout.addWidget(self.status)

        row = QHBoxLayout()
        self.btn_bgd = QPushButton("Take BGD")
        self.btn_flat = QPushButton("Take FLAT")
        self.btn_bpx = QPushButton("Take BADPIX")
        self.btn_bgd.clicked.connect(
            lambda: self._start("bgd", "Background", "masterbgd.fits"))
        self.btn_flat.clicked.connect(
            lambda: self._start("flat", "Flat", "masterflat.fits"))
        self.btn_bpx.clicked.connect(
            lambda: self._start("badpix", "Bad pixel map", "badpix.fits"))
        row.addWidget(self.btn_bgd)
        row.addWidget(self.btn_flat)
        row.addWidget(self.btn_bpx)
        layout.addLayout(row)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Output directory", self.outdir.text())
        if d:
            self.outdir.setText(d)

    def _load_camera(self, name):
        try:
            if name == "Sim":
                import fpwfsc.common.fake_hardware as fhw
                script_dir = Path(__file__).parent.absolute()
                hwconfig = script_dir.parent / "sim" / "sim_config.ini"
                hwspec = script_dir.parent / "sim" / "sim_config.spec"
                hw = sf.validate_config(str(hwconfig), str(hwspec))
                CSM = fhw.FakeCoronagraphOpticalSystem(**hw['SIMULATION']['OPTICAL_PARAMS'])
                self.camera = fhw.FakeDetector(opticalsystem=CSM,
                                               **hw['SIMULATION']['CAMERA_PARAMS'])
            elif name == "NIRC2":
                from fpwfsc.qacits import qacits_hardware
                self.camera = qacits_hardware.NIRC2Alias()
            else:
                raise ValueError(f"Unknown instrument: {name}")
            self.status.setText(f"Camera loaded: {name}")
        except Exception as e:
            self.camera = None
            self.status.setText(f"Camera load failed: {type(e).__name__}: {e}")

    def _start(self, kind, nice_name, filename):
        if self.camera is None:
            QMessageBox.warning(self, "No camera", "Camera not loaded.")
            return

        dlg = ConfirmDialog(nice_name, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return

        out = Path(self.outdir.text()).absolute()
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "Bad output dir", f"{out}\n{e}")
            return

        self._current_kind = kind
        self._current_filename = filename
        self._current_nice_name = nice_name
        n = self.n_images.value()
        self._set_buttons_enabled(False)
        self.status.setText(f"Acquiring 0/{n} for {nice_name}...")

        self.thread = AcquisitionThread(self.camera, n)
        self.thread.progress.connect(self._on_progress)
        self.thread.finished_ok.connect(self._on_acquired)
        self.thread.failed.connect(self._on_acq_failed)
        self.thread.start()

    def _on_progress(self, acquired, total):
        msg = f"Taking image {acquired}/{total} for {self._current_nice_name}"
        self.status.setText(msg + "...")
        print(msg)

    def _on_acquired(self, median_image):
        try:
            out = Path(self.outdir.text()).absolute()
            path = out / self._current_filename

            if self._current_kind == "badpix":
                # locate_badpix fits a Gaussian and sigma-clips; plot=True pops up
                # the matplotlib histogram the existing calibration script shows.
                data = sf.locate_badpix(median_image, sigmaclip=3, plot=True)
            elif self._current_kind == "flat":
                # Normalize by mean so equalize_image's `data/masterflat` stays
                # near the original scale.
                data = median_image / np.mean(median_image)
            else:
                data = median_image

            fits.PrimaryHDU(data).writeto(str(path), overwrite=True)
            try:
                os.chmod(str(path), 0o644)
            except OSError:
                pass
            self.status.setText(f"Saved {path}")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", f"{type(e).__name__}: {e}")
            self.status.setText(f"Save failed: {type(e).__name__}")
        finally:
            self._set_buttons_enabled(True)

    def _on_acq_failed(self, msg):
        QMessageBox.warning(self, "Acquisition failed", msg)
        self.status.setText("Acquisition failed")
        self._set_buttons_enabled(True)

    def _set_buttons_enabled(self, enabled):
        self.btn_bgd.setEnabled(enabled)
        self.btn_flat.setEnabled(enabled)
        self.btn_bpx.setEnabled(enabled)


def main():
    app = QApplication(sys.argv)
    w = CalibrationGUI()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
