#!/usr/bin/env python
import sys
import threading
import numpy as np
from collections import deque
import hcipy
from configobj import ConfigObj
import time
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import configparser
#from configupdater import ConfigUpdater

from fpwfsc.fnf import gui_helper as helper

from fpwfsc.common import plotting_funcs as pf
from fpwfsc.common import classes as ff_c
from fpwfsc.common import fake_hardware as fhw
from fpwfsc.common import support_functions as sf



from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QComboBox, QLineEdit, QPushButton, QFileDialog, QMessageBox,QGridLayout
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


def run_fastandfurious_test(camera='Sim', 
                            aosystem='Sim', 
                            FF_ini = 'FF_software_sim.ini', 
                            FF_spec = 'FF_software.spec', 
                            zernike_mode = 'Vertical Coma', 
                            amplitude = 1.5, 
                            flip_x = None,
                            flip_y = None,
                            rotation_angle_dm = None):
    
    settings = sf.validate_config(FF_ini, FF_spec)

    #----------------------------------------------------------------------
    # Control Loop parameters
    #----------------------------------------------------------------------
    gain              = settings['LOOP_SETTINGS']['gain']
    leak_factor       = settings['LOOP_SETTINGS']['leak factor']
    chosen_mode_basis = settings['LOOP_SETTINGS']['Used mode basis']
    Nmodes            = settings['LOOP_SETTINGS']['Number of modes']


    #----------------------------------------------------------------------
    # Optical model parameters
    #----------------------------------------------------------------------
    # Optical properties
    wavelength = settings['MODELLING']['wavelength (m)']
    mas_pix = settings['MODELLING']['pixel scale (mas/pix)']

    # Pupil and focal plane sampling
    Npix_pup = settings['MODELLING']['N pix pupil']
    Npix_foc = settings['MODELLING']['N pix focal']

    # Aperture and DM configuration
    chosen_aperture = settings['MODELLING']['aperture']
    rotation_angle_aperture = settings['MODELLING']['rotation angle aperture (deg)']
    if  rotation_angle_dm == None:
        rotation_angle_dm = settings['MODELLING']['rotation angle dm (deg)']

    # Image orientation settings
    if flip_x == None:
        flip_x = settings['MODELLING']['flip_x']
    if flip_x == None:
        flip_y = settings['MODELLING']['flip_y']


    #----------------------------------------------------------------------
    # F&F parameters
    #----------------------------------------------------------------------
    xcen                = settings['FF_SETTINGS']['xcen']
    ycen                = settings['FF_SETTINGS']['ycen']
    #WILBY SMOOTHIN IS NOT IMPLEMENTED YET!!!
    epsilon             = settings['FF_SETTINGS']['epsilon']
 
    #----------------------------------------------------------------------
    # Simulation parameters
    #----------------------------------------------------------------------
    flux                = settings['SIMULATION']['flux']
    exptime             = settings['SIMULATION']['exptime']
    rms_wfe             = settings['SIMULATION']['rms_wfe']
    seed                = settings['SIMULATION']['seed']
    #----------------------------------------------------------------------
    # Load the classes
    #----------------------------------------------------------------------
   


    Aperture = ff_c.Aperture(Npix_pup=Npix_pup,
                             aperturename=chosen_aperture,
                             rotation_angle_aperture=rotation_angle_aperture)

    OpticalModel = ff_c.SystemModel(aperture=Aperture,
                                    Npix_foc=Npix_foc,
                                    mas_pix=mas_pix,
                                    wavelength=wavelength)
    

    FnF = ff_c.FastandFurious(SystemModel=OpticalModel,
                              leak_factor=leak_factor,
                              gain=gain,
                              epsilon=epsilon,
                              chosen_mode_basis=chosen_mode_basis,
                              #apply_smoothing_filter=apply_smooth_filter,
                              number_of_modes=Nmodes)
    #----------------------------------------------------------------------
    # Load instruments
    #----------------------------------------------------------------------
    if camera == 'Sim' and aosystem == 'Sim':
        Camera = fhw.FakeDetector(opticalsystem=OpticalModel,
                                  flux = flux,
                                  exptime=exptime,
                                  dark_current_rate=0,
                                  read_noise=5,
                                  flat_field=0,
                                  include_photon_noise=False,
                                  xsize=1024,
                                  ysize=1024,
                                  field_center_x=330,
                                  field_center_y=430)

        AOsystem = fhw.FakeAOSystem(OpticalModel, modebasis=FnF.mode_basis,
                                    initial_rms_wfe=0, seed=seed,
                                    rotation_angle_dm = rotation_angle_dm,
                                    flip_x = flip_x,
                                    flip_y = flip_y)
    else:
        Camera, AOsystem = helper.load_instruments(camera,
                                                camargs={},
                                                aoargs={'rotation_angle_dm':
                                                        settings['MODELLING']['rotation angle dm (deg)'],
                                                        'flip_x':
                                                        settings['MODELLING']['flip_x'],
                                                        'flip_y':
                                                        settings['MODELLING']['flip_y']})

    zernike_name_to_index = {
    "Oblique Astigmatism": 5,
    "Vertical Astigmatism": 6,
    "Vertical Coma": 7,
    "Horizontal Coma": 8,
    "Vertical Trefoil": 9,
    "Oblique Trefoil": 10
    }
    zernike_index = zernike_name_to_index.get(zernike_mode)

    #create zernike mode
    mode_basis = hcipy.make_zernike_basis(num_modes = 1, D = 11.3, grid = Aperture.pupil_grid, starting_mode = zernike_index)
    mode_basis = sf.orthonormalize_mode_basis(mode_basis, Aperture.aperture)
    dm_volt_to_amp_amplify = 3



    for mode in mode_basis:
        
        # creating the phase that will be introduced
        phase_rad = mode * amplitude

        #bench image
        microns = phase_rad * FnF.wavelength / (2 * np.pi) * 1e6
        _,dm_microns = AOsystem.set_dm_data(microns*dm_volt_to_amp_amplify )
        image = Camera.take_image()
        image_bench = sf.reduce_images(image, xcen=xcen, ycen=ycen, npix=Npix_foc,
                                refpsf=OpticalModel.ref_psf.shaped,)

        #theory image
        pupil_wf = hcipy.Wavefront(Aperture.aperture * np.exp(1j * phase_rad),
                             wavelength=FnF.wavelength)
        focal_wf = OpticalModel.propagator(pupil_wf)
        image_theory = focal_wf.power


        fig, axs = plt.subplots(2, 2, figsize=(8, 8))


        the_img = hcipy.imshow_field(np.log10(image_theory / image_theory.max()), vmin=-3, ax=axs[0, 0])
        ben_img = axs[0,1].imshow(np.log10(np.abs(image_bench) / image_bench.max()), vmin=-3, origin='lower')
 
        the_DM = hcipy.imshow_field(phase_rad, cmap='bwr', vmin=-np.max(np.abs(phase_rad)), vmax=np.max(np.abs(phase_rad)),ax=axs[1, 0])
        
        dm_microns = dm_microns.reshape((21,21))
        max_bench = np.max(np.abs(dm_microns))
        ben_DM = axs[1,1].imshow(dm_microns, origin='lower', cmap='bwr',
                   vmin=-max_bench, vmax=max_bench)


        # revert to original cog file
        if aosystem == 'Sim':
            AOsystem.OpticalModel.update_pupil_wavefront(AOsystem.initial_phase_error)
        else:
            AOsystem.AO.revert_cog()
        
        return the_img, ben_img, the_DM, ben_DM
class FastAndFuriousGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DM Alignment for Fast and Furious")
        self.setMinimumWidth(400)

        # Example dropdown options ? update as needed
        self.camera_options = ["Sim", "NIRC2", "OSIRIS", "Palila", "Vampires"]
        self.aosystem_options = ["Sim", "KeckAO", "SCEXAO"]
        self.zernike_options = ["Oblique Astigmatism", "Vertical Astigmatism", "Vertical Coma", "Horizontal Coma", "Vertical Trefoil", "Oblique Trefoil"]


        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Camera dropdown
        self.camera_box = QComboBox()
        self.camera_box.addItems(self.camera_options)
        layout.addWidget(QLabel("Camera:"))
        layout.addWidget(self.camera_box)

        # AO System dropdown
        self.ao_box = QComboBox()
        self.ao_box.addItems(self.aosystem_options)
        layout.addWidget(QLabel("AO System:"))
        layout.addWidget(self.ao_box)

        # Zernike Mode dropdown
        self.zernike_box = QComboBox()
        self.zernike_box.addItems(self.zernike_options)
        layout.addWidget(QLabel("Zernike Mode:"))
        layout.addWidget(self.zernike_box)

        # Amplitude input
        self.amp_input = QLineEdit("1.5")
        layout.addWidget(QLabel("Amplitude:"))
        layout.addWidget(self.amp_input)

        # File selection for FF_ini
        self.FF_ini_path = 'FF_software_sim.ini'
        ini_layout = QHBoxLayout()
        self.ini_button = QPushButton("Select FF_ini File")
        self.ini_button.clicked.connect(self.select_ini_file)
        self.ini_label = QLabel('FF_software_sim.ini')
        ini_layout.addWidget(self.ini_button)
        ini_layout.addWidget(self.ini_label)
        layout.addLayout(ini_layout)

        self.save_button = QPushButton("Save config to FF_ini")
        self.save_button.clicked.connect(self.save_config_to_ini)
        layout.addWidget(self.save_button)

        # File selection for FF_spec
        self.FF_spec_path ='FF_software.spec'
        spec_layout = QHBoxLayout()
        self.spec_button = QPushButton("Select FF_spec File")
        self.spec_button.clicked.connect(self.select_spec_file)
        self.spec_label = QLabel('FF_software.spec')
        spec_layout.addWidget(self.spec_button)
        spec_layout.addWidget(self.spec_label)
        layout.addLayout(spec_layout)



        settings = sf.validate_config(self.FF_ini_path, self.FF_spec_path)
        flip_x = settings['MODELLING']['flip_x']
        flip_y = settings['MODELLING']['flip_y']
        rotation_angle_dm = settings['MODELLING']['rotation angle dm (deg)']
            
        # Flip X
        self.flip_x_checkbox = QCheckBox("Flip X")
        self.flip_x_checkbox.setChecked(flip_x)
        layout.addWidget(self.flip_x_checkbox)

        # Flip Y
        self.flip_y_checkbox = QCheckBox("Flip Y")
        self.flip_y_checkbox.setChecked(flip_y)
        layout.addWidget(self.flip_y_checkbox)

        # Rotation angle
        self.rotation_input = QLineEdit("0.0")
        layout.addWidget(QLabel("DM Rotation Angle (deg):"))
        self.rotation_input.setText(str(rotation_angle_dm))
        layout.addWidget(self.rotation_input)

        # Run button
        self.run_button = QPushButton("Run Alignment!")
        self.run_button.clicked.connect(self.run_test)
        layout.addWidget(self.run_button)

        self.setLayout(layout)
        self.init_plot_area()
    
    def init_plot_area(self):
        self.figures = []
        self.axes = []
        self.canvases = []

        self.plot_grid = QGridLayout()
        self.layout().addLayout(self.plot_grid)

        plotname = [['Theory Image','Bench Image'],['Theory DM Command', 'Applied DM Command']]

        for i in range(2):
            for j in range(2):
                fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
                canvas = FigureCanvas(fig)

                ax.set_title(plotname[i][j])
                ax.axis('off')  # Optional: turn off axis lines

                self.figures.append(fig)
                self.axes.append(ax)
                self.canvases.append(canvas)

                self.plot_grid.addWidget(canvas, i, j)

    def select_ini_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select FF_ini File")
        if file_name:
            self.FF_ini_path = file_name
            self.ini_label.setText(file_name.split("/")[-1])
            settings = sf.validate_config(self.FF_ini_path, self.FF_spec_path)
            flip_x = settings['MODELLING']['flip_x']
            flip_y = settings['MODELLING']['flip_y']
            rotation_angle_dm = settings['MODELLING']['rotation angle dm (deg)']
            self.flip_x_checkbox.setChecked(flip_x)
            self.flip_y_checkbox.setChecked(flip_y)
            self.rotation_input.setText(str(rotation_angle_dm))

    def select_spec_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select FF_spec File")
        if file_name:
            self.FF_spec_path = file_name
            self.spec_label.setText(file_name.split("/")[-1])

    
    def update_plots(self, images):
        for ax, canvas, img in zip(self.axes, self.canvases, images):
            current_title = ax.get_title()
            ax.clear()
            ax.imshow(img.get_array(), cmap=img.get_cmap(), norm=img.norm)
            ax.set_title(current_title)  # Clear or reset titles if needed
            ax.axis('off')
            canvas.draw()


    def save_config_to_ini(self):
        try:
            new_values = {
                'flip_x': self.flip_x_checkbox.isChecked(),
                'flip_y': self.flip_y_checkbox.isChecked(),
                'rotation angle dm (deg)': self.rotation_input.text(),
            }
            update_ini_file(self.FF_ini_path, new_values)
            QMessageBox.information(self, "Success", "INI file updated successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update INI file:\n{e}")


    def run_test(self):
        camera = self.camera_box.currentText()
        aosystem = self.ao_box.currentText()
        zernike_mode = self.zernike_box.currentText()
        try:
            amplitude = float(self.amp_input.text())
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Amplitude must be a number.")
            return

        if not self.FF_ini_path or not self.FF_spec_path:
            QMessageBox.warning(self, "Missing Files", "Please select both FF_ini and FF_spec files.")
            return
        
        try:
            rotation_angle_dm = float(self.rotation_input.text())
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Rotation angle must be a number.")
            return

        flip_x = self.flip_x_checkbox.isChecked()
        flip_y = self.flip_y_checkbox.isChecked()

        # Call your backend function
        the_img, ben_img, the_DM, ben_DM = run_fastandfurious_test(
            camera = camera, aosystem = aosystem, zernike_mode = zernike_mode, amplitude = amplitude,
            FF_ini = self.FF_ini_path, FF_spec = self.FF_spec_path,
            flip_x=flip_x, flip_y=flip_y, rotation_angle_dm=rotation_angle_dm
        )
        self.update_plots([the_img, ben_img, the_DM, ben_DM ])


# def update_ini_file(filepath, new_values, section='MODELLING'):
#     updater = ConfigUpdater()
#     updater.read(filepath)

#     # Update or add the specified keys
#     for key, value in new_values.items():
#         if updater[section].has_option(key):
#             updater[section][key].value = str(value)
#         else:
#             # Insert at the end of the section
#             updater[section].add_after(updater[section][-1], key, str(value))

#     # Write back to the same file
#     updater.update_file(filepath)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FastAndFuriousGUI()
    window.show()
    sys.exit(app.exec())
# if __name__ == "__main__":
#     run_fastandfurious_test(camera='Sim', 
#                             aosystem='Sim', 
#                             FF_ini = 'FF_software_sim.ini', 
#                             FF_spec = 'FF_software.spec',
#                             zernike_mode = 'Oblique Trefoil',
#                             amplitude = 1.5)