# import sys
# import threading
# import numpy as np
# from collections import deque
# import hcipy
# from configobj import ConfigObj
# import time
# import matplotlib.pyplot as plt
# from pathlib import Path
# import os
# import classes as ff_c
# import support_functions as sf

# from astropy.io import fits

# Aperture = ff_c.Aperture(Npix_pup=64,
#                              aperturename='open',
#                              rotation_angle_aperture=0)

# OpticalModel = ff_c.SystemModel(aperture=Aperture,
#                                     Npix_foc=64,
#                                     mas_pix=10,
#                                     wavelength=1550.0e-9)


# mode_basis = hcipy.make_zernike_basis(4, 11.3, Aperture.pupil_grid, 7)
# mode_basis = sf.orthonormalize_mode_basis(mode_basis, Aperture.aperture)

# mode = mode_basis[0]
# phase_rad = mode 

# pupil_wf = hcipy.Wavefront(Aperture.aperture * np.exp(1j * phase_rad),
#                              wavelength=1550.0e-9)
# focal_wf = OpticalModel.propagator(pupil_wf)
# image_theory = focal_wf.power


# hcipy.imshow_field(np.log10(image_theory / image_theory.max()), vmin=-3)
# plt.colorbar()
# plt.title('theory')

# plt.show()
# img = image_theory.shaped
# fits.writeto('COMA.fits',np.log10(img / img.max()), overwrite=True)

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QSlider, QFileDialog, QLabel, QPushButton
)
from PyQt5.QtCore import Qt
from astropy.io import fits
from scipy.ndimage import rotate
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from astropy.visualization import ZScaleInterval



class FitsViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS Viewer with Rotation Slider")

        self.image_data = None
        self.original_data = None

        self.canvas = FigureCanvas(Figure())
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.axis('off')

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(360)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_rotation)

        self.load_button = QPushButton("Load FITS File")
        self.load_button.clicked.connect(self.load_fits)

        self.slider_label = QLabel("Rotation: 0°")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.slider_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.load_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_fits(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open FITS File", "", "FITS files (*.fits *.fit)")
        if file_path:
            with fits.open(file_path) as hdul:
                self.original_data = hdul[0].data.astype(float)
            self.slider.setValue(0)
            self.update_display(self.original_data)

    def update_display(self, data):
        self.ax.clear()
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(data)
        self.ax.imshow(data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        self.ax.axis('off')
        self.canvas.draw()


    def update_rotation(self, value):
        if self.original_data is None:
            return
        self.slider_label.setText(f"Rotation: {value}°")
        rotated = rotate(self.original_data, -value, reshape=False, order=3)
        self.update_display(rotated)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = FitsViewer()
    viewer.resize(800, 600)
    viewer.show()
    sys.exit(app.exec_())



