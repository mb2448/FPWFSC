import sys
import os
import queue  # For thread-safe setpoint updates

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QComboBox, QPushButton,
                             QScrollArea, QFrame, QToolButton, QSizePolicy,
                             QFileDialog, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QParallelAnimationGroup, QPropertyAnimation, QAbstractAnimation, QTimer, QThread
from PyQt5.QtGui import QFont

from configobj import ConfigObj, ConfigObjError, flatten_errors
import threading
from pathlib import Path

# Import the qacits GUI helper module
import fpwfsc.qacits.qacits_gui_helper as helper

# Import the qacits plotter
from fpwfsc.qacits.qacits_plotter_qt import QacitsPlotter

# Import the run function from run_qacits.py
from fpwfsc.qacits.run_qacits import run as original_run

# Import the custom validator
from fpwfsc.common.support_functions import MyValidator


class AlgorithmThread(QThread):
    """Thread to run the qacits tracking algorithm with proper stop handling"""
    
    # Signal to update the plot from the main thread
    plot_update_signal = pyqtSignal(object, float, float, float, float, object, object, str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, camera, aosystem, config, spec_file, my_event, main_window,
                 param_update_queue, centroid_offset_queue, centroid_offset_feedback,
                 warning_queue):
        super().__init__()
        self.camera = camera
        self.aosystem = aosystem
        self.config = config
        self.spec_file = spec_file
        self.my_event = my_event
        self.main_window = main_window
        self.param_update_queue = param_update_queue
        self.centroid_offset_queue = centroid_offset_queue
        self.centroid_offset_feedback = centroid_offset_feedback
        self.warning_queue = warning_queue

    def run(self):
        try:
            original_run(camera=self.camera,
                         aosystem=self.aosystem,
                         config=self.config,
                         configspec=self.spec_file,
                         my_event=self.my_event,
                         plotter=None,
                         plot_signal=self.plot_update_signal if self.main_window.plotter else None,
                         param_update_queue=self.param_update_queue,
                         centroid_offset_queue=self.centroid_offset_queue,
                         centroid_offset_feedback=self.centroid_offset_feedback,
                         warning_queue=self.warning_queue)
        except Exception as e:
            import traceback
            print(f"Error in algorithm thread: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            self.error_signal.emit(str(e))
        finally:
            print("Algorithm thread finished")


class CollapsibleBox(QWidget):
    """A collapsible box widget for expert options"""
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.clicked.connect(self.on_clicked)

        self.content_area = QScrollArea()
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setWidgetResizable(True)
        self.content_area.setFrameShape(QFrame.NoFrame)
        self.content_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.content_widget = QWidget()
        self.content_area.setWidget(self.content_widget)

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation = QParallelAnimationGroup(self)
        self.toggle_animation.addAnimation(QPropertyAnimation(self, b"minimumHeight"))
        self.toggle_animation.addAnimation(QPropertyAnimation(self, b"maximumHeight"))
        self.toggle_animation.addAnimation(QPropertyAnimation(self.content_area, b"maximumHeight"))

        self.toggle_animation.finished.connect(self.animation_finished)

        self.is_animating = False
        self.is_expanded = False

    @pyqtSlot()
    def on_clicked(self):
        """Handle toggle button click"""
        if self.is_animating:
            return

        self.is_animating = True
        self.toggle_button.setEnabled(False)

        if not self.is_expanded:
            self.toggle_button.setArrowType(Qt.DownArrow)
            self.toggle_animation.setDirection(QAbstractAnimation.Forward)
        else:
            self.toggle_button.setArrowType(Qt.RightArrow)
            self.toggle_animation.setDirection(QAbstractAnimation.Backward)

        self.toggle_animation.start()

    def animation_finished(self):
        """Handle animation finished event"""
        self.is_animating = False
        self.toggle_button.setEnabled(True)
        self.is_expanded = not self.is_expanded

    def setContentLayout(self, layout):
        """Set the layout of the content area"""
        self.content_widget.setLayout(layout)
        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        content_height = layout.sizeHint().height()

        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(300)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(self.toggle_animation.animationCount() - 1)
        content_animation.setDuration(300)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)


class QacitsConfigGUI(QWidget):
    """Main GUI class for the qacits configuration editor"""
    def __init__(self):
        super().__init__()

        # Get file paths
        script_dir = Path(__file__).parent
        config_path = script_dir/"qacits_config.ini"
        spec_path = script_dir/"qacits_config.spec"

        self.config_file = str(config_path)
        self.spec_file = str(spec_path)
        self.is_running = False
        self.my_event = threading.Event()
        self.plotter = None
        self._stopping = False
        
        # Queue for parameter updates to the running loop
        self.param_update_queue = queue.Queue()

        # Queue for centroid offset updates to the running loop
        self.centroid_offset_queue = queue.Queue()
        self._capture_on_first_iteration = False
        # Queue for feedback from the loop after applying an offset
        self.centroid_offset_feedback = queue.Queue()
        # Queue for non-fatal warnings from the loop
        self.warning_queue = queue.Queue()

        # References to setpoint input widgets (set during create_widgets)
        self.x_setpoint_widget = None
        self.y_setpoint_widget = None
        self.x_centroid_offset_widget = None
        self.y_centroid_offset_widget = None

        self.initUI()

    def closeEvent(self, event):
        """Handle window close event - ensure all resources are cleaned up"""
        print("Window closing...")

        self.thread_check_timer.stop()

        if hasattr(self, 'my_event'):
            self.my_event.set()

        if hasattr(self, 'algorithm_thread') and self.algorithm_thread.isRunning():
            try:
                self.algorithm_thread.plot_update_signal.disconnect()
            except Exception:
                pass
            print("Waiting for algorithm thread to finish...")
            self.algorithm_thread.wait(3000)

        event.accept()
        QApplication.quit()

    def initUI(self):
        self.setWindowTitle('miniQACITS')
        self.resize(325, 610)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Hardware selection
        hardware_widget = QWidget()
        hardware_layout = QHBoxLayout(hardware_widget)
        hardware_layout.setContentsMargins(0, 0, 0, 0)

        hardware_label = QLabel("Hardware")
        hardware_layout.addWidget(hardware_label)

        self.hardware_select = QComboBox()
        self.hardware_select.addItems(helper.valid_instruments)
        self.hardware_select.currentTextChanged.connect(self.on_hardware_changed)
        self.hardware_select.setFixedHeight(20)
        hardware_layout.addWidget(self.hardware_select)

        main_layout.addWidget(hardware_widget)

        # Scrollable config area
        scroll = QScrollArea(self)
        main_layout.addWidget(scroll)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_content = QWidget(scroll)

        self.config_layout = QVBoxLayout(scroll_content)
        self.config_layout.setSpacing(2)
        self.config_layout.setContentsMargins(5, 5, 5, 5)
        scroll.setWidget(scroll_content)

        self.load_config(initial_load=True)
        self.create_widgets()

        main_layout.addSpacing(10)

        # Buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        config_buttons_layout = QHBoxLayout()
        config_buttons_layout.setSpacing(10)

        save_button = QPushButton('Save configuration')
        save_button.clicked.connect(self.save_config)
        config_buttons_layout.addWidget(save_button)

        load_config_button = QPushButton('Load configuration')
        load_config_button.clicked.connect(lambda: self.load_config(None))
        config_buttons_layout.addWidget(load_config_button)

        button_layout.addLayout(config_buttons_layout)
        button_layout.addSpacing(5)

        self.update_params_button = QPushButton('Update Loop Parameters')
        self.update_params_button.setFont(QFont('Arial', 12, QFont.Bold))
        self.update_params_button.clicked.connect(self.on_update_params_clicked)
        self.update_params_button.setEnabled(False)
        self.update_params_button.setStyleSheet("background-color: #4CAF50; color: white;")
        button_layout.addWidget(self.update_params_button)

        button_layout.addSpacing(5)

        self.run_stop_button = QPushButton('Run')
        self.run_stop_button.setFont(QFont('Arial', 12, QFont.Bold))
        self.run_stop_button.clicked.connect(self.toggle_run_stop)
        self.run_stop_button.setStyleSheet("background-color: green; color: white;")
        button_layout.addWidget(self.run_stop_button)

        button_layout.addSpacing(5)

        # Bottom row: Reset DTT + Take Test Image side by side
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(5)

        self.reset_dtt_button = QPushButton('Reset DTT Offset')
        self.reset_dtt_button.setFont(QFont('Arial', 12, QFont.Bold))
        self.reset_dtt_button.clicked.connect(self.on_reset_dtt_clicked)
        self.reset_dtt_button.setStyleSheet("background-color: #FF9800; color: white;")
        bottom_row.addWidget(self.reset_dtt_button)

        self.test_image_button = QPushButton('Take Test Image')
        self.test_image_button.setFont(QFont('Arial', 12, QFont.Bold))
        self.test_image_button.clicked.connect(self.on_test_image_clicked)
        self.test_image_button.setStyleSheet("background-color: #607D8B; color: white;")
        bottom_row.addWidget(self.test_image_button)

        button_layout.addLayout(bottom_row)

        main_layout.addLayout(button_layout)

        self.thread_check_timer = QTimer()
        self.thread_check_timer.timeout.connect(self.check_thread_status)
        self.thread_check_timer.start(1000)

        self.default_hardware = helper.valid_instruments[0]
        self.on_hardware_changed(self.default_hardware)

    def on_hardware_changed(self, selected_hardware):
        """Handle changes to the hardware selection dropdown"""
        try:
            print(f"Loading {selected_hardware}...")
            self.camera, self.aosystem = helper.load_instruments(selected_hardware)
            print(f"{selected_hardware} loaded successfully")
        except Exception as e:
            print(f"Error loading {selected_hardware}: {str(e)}")
            print("Loading Sim as default")
            default_index = self.hardware_select.findText('Sim')
            if default_index >= 0:
                self.hardware_select.blockSignals(True)
                self.hardware_select.setCurrentIndex(default_index)
                self.hardware_select.blockSignals(False)
                self.camera, self.aosystem = helper.load_instruments('Sim')

    def on_update_params_clicked(self):
        """Push all current GUI values to the running loop."""
        try:
            self.update_config_from_gui()
            params = {
                'n_iter': int(self.config['EXECUTION']['N iterations']),
                'setpointx': float(self.config['EXECUTION']['x setpoint']),
                'setpointy': float(self.config['EXECUTION']['y setpoint']),
                'inner_rad': int(self.config['EXECUTION']['inner radius']),
                'outer_rad': int(self.config['EXECUTION']['outer radius']),
                'Kp': float(self.config['PID']['Kp']),
                'Ki': float(self.config['PID']['Ki']),
                'Kd': float(self.config['PID']['Kd']),
                'output_limits': float(self.config['PID']['output_limits']),
                'tt_gain': float(self.config['AO']['tip tilt gain']),
                'tt_rot_deg': float(self.config['AO']['tip tilt angle (deg)']),
                'tt_flipx': str(self.config['AO']['tip tilt flip x']).lower() == 'true',
                'tt_flipy': str(self.config['AO']['tip tilt flip y']).lower() == 'true',
                'centroid_offset_x': float(self.config['PID']['x centroid offset']),
                'centroid_offset_y': float(self.config['PID']['y centroid offset']),
                'background_file': self.config['CAMERA CALIBRATION']['background file'],
                'masterflat_file': self.config['CAMERA CALIBRATION']['masterflat file'],
                'badpix_file': self.config['CAMERA CALIBRATION']['badpix file'],
            }
            self.param_update_queue.put(params)
            print(f"Loop parameters queued for update")
        except (ValueError, TypeError) as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid parameter", str(e))

    def on_capture_offset_clicked(self):
        """Tell the running loop to capture the current quad-cell reading as the PID setpoint.
        Can be clicked before Run to arm capture on the first iteration."""
        if self.is_running:
            self.centroid_offset_queue.put(None)
        else:
            self._capture_on_first_iteration = True
        self.capture_offset_button.setText("Capturing...")
        self.capture_offset_button.setEnabled(False)
        self.capture_offset_button.setStyleSheet("background-color: #90CAF9; color: white;")

    def on_zero_offset_clicked(self):
        """Reset centroid offset to (0, 0) — return to normal centering."""
        self.x_centroid_offset_widget.setText("0.0")
        self.y_centroid_offset_widget.setText("0.0")
        if self.is_running:
            self.centroid_offset_queue.put((0.0, 0.0))
        print("Centroid offset zeroed")

    def on_test_image_clicked(self):
        """Take a single image and display the cropped region around the setpoint."""
        self.test_image_button.setText("Taking...")
        self.test_image_button.setEnabled(False)
        QApplication.processEvents()

        try:
            from pathlib import Path
            import fpwfsc.common.support_functions as sf
            import fpwfsc.qacits.qacits_funcs as qf

            # Get or build camera
            if self.camera == 'Sim':
                import fpwfsc.common.fake_hardware as fhw
                sim_dir = Path(__file__).parent.absolute().parent / "sim"
                hw = sf.validate_config(str(sim_dir / "sim_config.ini"),
                                        str(sim_dir / "sim_config.spec"))
                CSM = fhw.FakeCoronagraphOpticalSystem(**hw['SIMULATION']['OPTICAL_PARAMS'])
                cam = fhw.FakeDetector(opticalsystem=CSM, **hw['SIMULATION']['CAMERA_PARAMS'])
            else:
                cam = self.camera

            # Read config from GUI
            self.update_config_from_gui()
            setpointx = float(self.config['EXECUTION']['x setpoint'])
            setpointy = float(self.config['EXECUTION']['y setpoint'])
            inner_rad = int(self.config['EXECUTION']['inner radius'])
            outer_rad = int(self.config['EXECUTION']['outer radius'])

            bgds = {
                'bkgd':       sf.load_fits_or_none(self.config['CAMERA CALIBRATION']['background file']),
                'masterflat': sf.load_fits_or_none(self.config['CAMERA CALIBRATION']['masterflat file']),
                'badpix':     sf.load_fits_or_none(self.config['CAMERA CALIBRATION']['badpix file']),
            }

            # Take image, calibrate, crop
            raw = cam.take_image()
            calibrated = sf.equalize_image(raw, **bgds)

            xs, ys, cropped = qf.crop_to_square(calibrated, cx=setpointx, cy=setpointy,
                                                 size=outer_rad * 2 + 1)

            # Display in a popup plotter
            from fpwfsc.qacits.qacits_plotter_qt import QacitsPlotter
            self._test_plotter = QacitsPlotter(figsize=(400, 400))
            self._test_plotter.setWindowTitle("Take Test Image")
            self._test_plotter.update(image=cropped, x_center=setpointx, y_center=setpointy,
                                      min_radius=inner_rad, max_radius=outer_rad,
                                      x_coords=xs, y_coords=ys, title="Take Test Image")

        except Exception as e:
            print(f"Test image error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.test_image_button.setText("Take Test Image")
            self.test_image_button.setEnabled(True)

    def on_reset_dtt_clicked(self):
        """Handle reset DTT offset button click"""
        if self.aosystem is None or isinstance(self.aosystem, str):
            print("No AO system loaded (sim mode). Cannot reset DTT offset.")
            return
        try:
            self.aosystem.zero_tiptilt()
            print("DTT offset reset to zero.")
        except Exception as e:
            print(f"Error resetting DTT offset: {e}")

    def check_thread_status(self):
        """Check if algorithm thread is still running, and poll for feedback/warnings"""
        # Check for warnings from the loop
        try:
            msg = self.warning_queue.get_nowait()
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Warning", msg)
        except queue.Empty:
            pass

        # Check for centroid offset feedback from the loop
        try:
            xo, yo = self.centroid_offset_feedback.get_nowait()
            self.x_centroid_offset_widget.setText(f"{xo:.4f}")
            self.y_centroid_offset_widget.setText(f"{yo:.4f}")
            self.capture_offset_button.setText("Capture Offset")
            self.capture_offset_button.setEnabled(True)
            self.capture_offset_button.setStyleSheet("background-color: #2196F3; color: white;")
        except queue.Empty:
            pass

        if hasattr(self, 'algorithm_thread'):
            if not self.algorithm_thread.isRunning() and (self.is_running or self._stopping):
                self.is_running = False
                self.run_stop_button.setText('Run')
                self.run_stop_button.setStyleSheet("background-color: green; color: white;")
                self.update_params_button.setEnabled(False)
                # Keep capture_offset_button enabled so it can be pre-armed
                self.capture_offset_button.setEnabled(True)
                self.capture_offset_button.setText("Capture Offset")
                self.capture_offset_button.setStyleSheet("background-color: #2196F3; color: white;")
                self.cleanup_resources()

    @pyqtSlot(str)
    def handle_thread_error(self, message):
        """Show algorithm thread errors as a popup in the GUI."""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Algorithm Error", message)

    @pyqtSlot(object, float, float, float, float, object, object, str)
    def handle_plot_update(self, image, x_center, y_center, min_radius, max_radius, x_coords, y_coords, title):
        """Handle plot update signal from worker thread - runs in main thread"""
        if self.plotter is not None and not self.plotter.closed:
            self.plotter.update(image=image, x_center=x_center, y_center=y_center,
                               min_radius=min_radius, max_radius=max_radius,
                               x_coords=x_coords, y_coords=y_coords, title=title)

    def toggle_run_stop(self):
        """Handle Run/Stop button press"""
        if not self.is_running:
            self.is_running = True
            self.run_stop_button.setText('Stop')
            self.run_stop_button.setStyleSheet("background-color: red; color: white;")
            self.update_params_button.setEnabled(True)
            self.capture_offset_button.setEnabled(True)
            print("Running")
            self.my_event = threading.Event()

            # Clear queues before starting
            for q in (self.param_update_queue, self.centroid_offset_queue, self.centroid_offset_feedback, self.warning_queue):
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

            # Re-arm capture if it was pre-armed before Run
            if self._capture_on_first_iteration:
                self.centroid_offset_queue.put(None)
                self._capture_on_first_iteration = False

            # Update config from GUI before starting
            self.update_config_from_gui()

            self.run_config()
        else:
            self.is_running = False
            self.run_stop_button.setText('Run')
            self.run_stop_button.setStyleSheet("background-color: green; color: white;")
            self.run_stop_button.setEnabled(False)
            self.update_params_button.setEnabled(False)
            # Keep capture_offset_button enabled for pre-arming
            print("Stopping")

            if hasattr(self, '_stopping') and self._stopping:
                print("Already attempting to stop, please wait...")
                return

            self._stopping = True

            if hasattr(self, 'my_event'):
                print("Setting stop event — waiting for current iteration to finish")
                self.my_event.set()
                # check_thread_status (1-second timer) will detect when the
                # thread exits and call cleanup_resources to re-enable buttons.

    def cleanup_resources(self):
        """Clean up any resources that need to be explicitly closed"""
        if hasattr(self, '_stopping'):
            self._stopping = False

        self.run_stop_button.setEnabled(True)

    def load_config(self, file_name=None, initial_load=False):
        """Load configuration file and update GUI"""
        if initial_load:
            file_name = self.config_file
        elif file_name is None:
            dialog = QFileDialog(self)
            dialog.setNameFilter("INI Files (*.ini)")
            dialog.setFileMode(QFileDialog.ExistingFile)
            dialog.setViewMode(QFileDialog.List)

            if dialog.exec_():
                file_name = dialog.selectedFiles()[0]
            else:
                print("No file selected.")
                return

        try:
            new_config = ConfigObj(file_name, configspec=self.spec_file)
            validator = MyValidator()
            results = new_config.validate(validator, preserve_errors=True)

            if results is not True:
                error_messages = []
                for section_list, key, error in flatten_errors(new_config, results):
                    if key is not None:
                        section_str = '.'.join(section_list)
                        if error is False:
                            error_messages.append(f"Missing value or section for '{section_str}.{key}'")
                        else:
                            error_messages.append(f"Invalid value for '{section_str}.{key}': {error}")
                    else:
                        error_messages.append(f"Missing section: {'.'.join(section_list)}")

                if error_messages:
                    print("Configuration validation failed:")
                    for msg in error_messages:
                        print(f"  - {msg}")
                    raise ConfigObjError("Config validation failed")

            self.config = new_config
            self.update_gui_from_config()
            print(f"Configuration loaded successfully from {file_name}")
        except (IOError, ConfigObjError) as e:
            print(f"Error loading configuration: {e}")
            if initial_load:
                print("Error loading default configuration. Creating config from spec file.")
                self.config = ConfigObj(configspec=self.spec_file)
                validator = MyValidator()
                result = self.config.validate(validator, copy=True)

                if result is not True:
                    print("Warning: Some validation errors occurred with defaults")
                    for section_list, key, error in flatten_errors(self.config, result):
                        if key is not None:
                            section_str = '.'.join(section_list)
                            if error is False:
                                print(f"Missing value for '{section_str}.{key}'")
                            else:
                                print(f"Invalid value for '{section_str}.{key}': {error}")
                        else:
                            print(f"Missing section: {'.'.join(section_list)}")

                print("Created configuration using defaults from spec file")

                self.update_gui_from_config()

    def create_widgets(self):
        """Create widgets for each configuration section"""
        while self.config_layout.count():
            item = self.config_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for section, items in self.config.items():
            section_frame = QFrame()
            section_frame.setFrameShape(QFrame.StyledPanel)
            section_layout = QVBoxLayout(section_frame)
            section_layout.setSpacing(2)
            section_layout.setContentsMargins(5, 5, 5, 5)

            section_label = QLabel(section)
            bold_font = section_label.font()
            bold_font.setBold(True)
            section_label.setFont(bold_font)
            section_layout.addWidget(section_label)

            regular_options = []
            expert_options = []

            for key, value in items.items():
                if helper.is_expert_option(section, key):
                    expert_options.append((key, value))
                else:
                    regular_options.append((key, value))

            for key, value in regular_options:
                # Skip items rendered custom below
                if section == 'EXECUTION' and key in ('x setpoint', 'y setpoint', 'log', 'logdir'):
                    continue
                if section == 'PID' and key in ('x centroid offset', 'y centroid offset'):
                    continue
                item_widget = self.create_item_widget(key, value, section)
                section_layout.addWidget(item_widget)

            # Custom rows for EXECUTION section
            if section == 'EXECUTION':
                # Log row: log [...][dir text][True/False] — right under plot
                log_row = QWidget()
                log_row.setFixedHeight(24)
                log_layout = QHBoxLayout(log_row)
                log_layout.setContentsMargins(0, 0, 0, 0)
                log_layout.setSpacing(3)
                log_layout.addWidget(QLabel("log"))

                log_browse = QPushButton()
                log_browse.setIcon(self.style().standardIcon(self.style().SP_DirOpenIcon))
                log_browse.setFixedSize(22, 22)
                self.logdir_widget = QLineEdit(str(self.config['EXECUTION']['logdir']))
                self.logdir_widget.setFixedHeight(20)
                self.logdir_widget.setMaximumWidth(140)
                self.logdir_widget.setReadOnly(True)
                self.logdir_widget.setToolTip(helper.get_help_message('EXECUTION', 'logdir'))

                def browse_logdir(tf=self.logdir_widget):
                    d = QFileDialog.getExistingDirectory(None, "Log directory", tf.text())
                    if d:
                        tf.setText(d)
                log_browse.clicked.connect(lambda _: browse_logdir())

                log_layout.addWidget(log_browse)
                log_layout.addWidget(self.logdir_widget)

                self.log_toggle_widget = QComboBox()
                self.log_toggle_widget.addItems(['True', 'False'])
                self.log_toggle_widget.setCurrentText(str(self.config['EXECUTION']['log']))
                self.log_toggle_widget.setFixedHeight(20)
                self.log_toggle_widget.setToolTip(helper.get_help_message('EXECUTION', 'log'))
                log_layout.addWidget(self.log_toggle_widget)

                section_layout.addWidget(log_row)

                # Setpoint row: setpoint: x:[ ] y:[ ]
                sp_row = QWidget()
                sp_layout = QHBoxLayout(sp_row)
                sp_layout.setContentsMargins(0, 0, 0, 0)
                sp_layout.addWidget(QLabel("setpoint:"))
                sp_layout.addWidget(QLabel("x:"))
                self.x_setpoint_widget = QLineEdit(str(self.config['EXECUTION']['x setpoint']))
                self.x_setpoint_widget.setFixedHeight(20)
                self.x_setpoint_widget.setFixedWidth(60)
                self.x_setpoint_widget.setToolTip(helper.get_help_message('EXECUTION', 'x setpoint'))
                sp_layout.addWidget(self.x_setpoint_widget)
                sp_layout.addWidget(QLabel("y:"))
                self.y_setpoint_widget = QLineEdit(str(self.config['EXECUTION']['y setpoint']))
                self.y_setpoint_widget.setFixedHeight(20)
                self.y_setpoint_widget.setFixedWidth(60)
                self.y_setpoint_widget.setToolTip(helper.get_help_message('EXECUTION', 'y setpoint'))
                sp_layout.addWidget(self.y_setpoint_widget)
                sp_layout.addStretch()
                section_layout.addWidget(sp_row)

            # Custom centroid offset row + Capture Offset button for PID section
            if section == 'PID':
                # centroid offset: x:[  ] y:[  ]
                offset_row = QWidget()
                offset_layout = QHBoxLayout(offset_row)
                offset_layout.setContentsMargins(0, 0, 0, 0)
                offset_layout.addWidget(QLabel("centroid offset:"))
                offset_layout.addWidget(QLabel("x:"))
                self.x_centroid_offset_widget = QLineEdit(str(self.config['PID']['x centroid offset']))
                self.x_centroid_offset_widget.setFixedHeight(20)
                self.x_centroid_offset_widget.setFixedWidth(60)
                self.x_centroid_offset_widget.setToolTip(helper.get_help_message('PID', 'x centroid offset'))
                offset_layout.addWidget(self.x_centroid_offset_widget)
                offset_layout.addWidget(QLabel("y:"))
                self.y_centroid_offset_widget = QLineEdit(str(self.config['PID']['y centroid offset']))
                self.y_centroid_offset_widget.setFixedHeight(20)
                self.y_centroid_offset_widget.setFixedWidth(60)
                self.y_centroid_offset_widget.setToolTip(helper.get_help_message('PID', 'y centroid offset'))
                offset_layout.addWidget(self.y_centroid_offset_widget)
                offset_layout.addStretch()
                section_layout.addWidget(offset_row)

                section_layout.addSpacing(4)

                # Two half-width buttons side by side
                btn_row = QWidget()
                btn_layout = QHBoxLayout(btn_row)
                btn_layout.setContentsMargins(0, 0, 0, 0)
                btn_layout.setSpacing(5)

                self.capture_offset_button = QPushButton('Capture Offset')
                self.capture_offset_button.setFont(QFont('Arial', 12, QFont.Bold))
                self.capture_offset_button.clicked.connect(self.on_capture_offset_clicked)
                self.capture_offset_button.setStyleSheet("background-color: #2196F3; color: white;")
                btn_layout.addWidget(self.capture_offset_button)

                self.zero_offset_button = QPushButton('Zero Offset')
                self.zero_offset_button.setFont(QFont('Arial', 12, QFont.Bold))
                self.zero_offset_button.clicked.connect(self.on_zero_offset_clicked)
                self.zero_offset_button.setStyleSheet("background-color: #9E9E9E; color: white;")
                btn_layout.addWidget(self.zero_offset_button)

                section_layout.addWidget(btn_row)
                section_layout.addSpacing(4)

            if expert_options:
                expert_box = CollapsibleBox("Expert Options")
                expert_layout = QGridLayout()
                expert_layout.setVerticalSpacing(2)
                expert_layout.setHorizontalSpacing(5)
                for i, (key, value) in enumerate(expert_options):
                    label = QLabel(key)
                    input_widget = self.create_input_widget(section, key, value)
                    expert_layout.addWidget(label, i, 0)
                    expert_layout.addWidget(input_widget, i, 1)

                    tooltip = helper.get_help_message(section, key)
                    label.setToolTip(tooltip)
                    input_widget.setToolTip(tooltip)

                expert_box.setContentLayout(expert_layout)
                section_layout.addWidget(expert_box)

            if len(regular_options) <= 2 and len(expert_options) == 0:
                section_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

            self.config_layout.addWidget(section_frame)

    def create_item_widget(self, key, value, section):
        """Create a widget for a single configuration item"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(key)
        label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        layout.addWidget(label)

        input_widget = self.create_input_widget(section, key, value)
        layout.addWidget(input_widget)

        tooltip = helper.get_help_message(section, key)
        label.setToolTip(tooltip)
        input_widget.setToolTip(tooltip)
        
        return widget

    def create_input_widget(self, section, key, value):
        """Create an appropriate input widget based on the configuration specification"""
        spec = self.get_spec_for_key(f"{section}.{key}")

        if helper.is_file_option(section, key):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            text_field = QLineEdit(str(value))
            text_field.setFixedHeight(20)
            text_field.setPlaceholderText("(none)")
            text_field.setReadOnly(True)
            text_field.textChanged.connect(lambda text, tf=text_field: self._validate_file_path(tf))
            self._validate_file_path(text_field)  # check initial value

            browse_button = QPushButton()
            browse_button.setIcon(self.style().standardIcon(self.style().SP_FileIcon))
            browse_button.setFixedSize(22, 22)

            def browse_file(tf=text_field):
                start_path = tf.text() if tf.text() else str(Path.home())
                file_path, _ = QFileDialog.getOpenFileName(
                    None, "Select FITS File", start_path,
                    "FITS Files (*.fits *.fit *.FITS *.FIT);;All Files (*)"
                )
                if file_path:
                    tf.setText(file_path)

            browse_button.clicked.connect(lambda _: browse_file())

            layout.addWidget(text_field, 1)
            layout.addWidget(browse_button, 0)

            widget.text_field = text_field
            widget.setFixedHeight(22)
            return widget

        elif helper.is_directory_option(section, key):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            text_field = QLineEdit(str(value))
            text_field.setFixedHeight(20)
            text_field.setReadOnly(True)

            browse_button = QPushButton()
            browse_button.setIcon(self.style().standardIcon(self.style().SP_DirOpenIcon))
            browse_button.setFixedSize(22, 22)

            def browse_directory(tf=text_field):
                directory = QFileDialog.getExistingDirectory(
                    None, "Select Directory", tf.text(),
                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
                )
                if directory:
                    tf.setText(directory)

            browse_button.clicked.connect(lambda _: browse_directory())

            layout.addWidget(text_field, 1)
            layout.addWidget(browse_button, 0)

            widget.text_field = text_field
            return widget

        elif spec and 'boolean' in spec:
            input_widget = QComboBox()
            input_widget.addItems(['True', 'False'])
            input_widget.setCurrentText(str(value))
            input_widget.setFixedHeight(20)
            return input_widget
        elif spec and 'option(' in spec:
            input_widget = QComboBox()
            options = spec.split('option(')[1].split(')')[0].replace("'", "").split(',')
            input_widget.addItems([opt.strip() for opt in options])
            input_widget.setCurrentText(str(value).strip())
            input_widget.setFixedHeight(20)
            return input_widget
        else:
            input_widget = QLineEdit(str(value))
            input_widget.setFixedHeight(20)
            return input_widget

    def get_spec_for_key(self, key):
        """Get the specification for a given key from the config spec"""
        keys = key.split('.')
        spec = self.config.configspec
        for k in keys:
            if k in spec:
                spec = spec[k]
            else:
                return None
        return spec

    def save_config(self):
        """Save the current configuration to a file"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        default_name = "qacits_config.ini"
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            default_name,
            "INI Files (*.ini);;All Files (*)",
            options=options
        )
        if file_name:
            if not file_name.lower().endswith('.ini'):
                file_name += '.ini'
            self.update_config_from_gui()
            self.config.filename = file_name
            self.config.write()
            print(f"Configuration saved successfully to {file_name}")

    def update_config_from_gui(self):
        """Update the configuration object with values from GUI widgets"""
        for i in range(self.config_layout.count()):
            section_frame = self.config_layout.itemAt(i).widget()
            if isinstance(section_frame, QFrame):
                section_layout = section_frame.layout()
                section_label = section_layout.itemAt(0).widget()
                section = section_label.text()

                for j in range(1, section_layout.count()):
                    item = section_layout.itemAt(j).widget()
                    if isinstance(item, QWidget):
                        if isinstance(item, CollapsibleBox):
                            content_widget = item.content_area.widget()
                            if content_widget and content_widget.layout():
                                content_layout = content_widget.layout()
                                for k in range(content_layout.rowCount()):
                                    label_item = content_layout.itemAtPosition(k, 0)
                                    input_item = content_layout.itemAtPosition(k, 1)
                                    if label_item and input_item:
                                        key = label_item.widget().text()
                                        value = self.get_widget_value(input_item.widget())
                                        self.config[section][key] = value
                        else:
                            item_layout = item.layout()
                            if item_layout and item_layout.count() == 2:
                                first_widget = item_layout.itemAt(0).widget()
                                if isinstance(first_widget, QLabel):
                                    key = first_widget.text()
                                    value = self.get_widget_value(item_layout.itemAt(1).widget())
                                    self.config[section][key] = value

        # Custom inline widgets not found by the generic walker
        if self.x_setpoint_widget is not None:
            self.config['EXECUTION']['x setpoint'] = self.x_setpoint_widget.text()
        if self.y_setpoint_widget is not None:
            self.config['EXECUTION']['y setpoint'] = self.y_setpoint_widget.text()
        if self.x_centroid_offset_widget is not None:
            self.config['PID']['x centroid offset'] = self.x_centroid_offset_widget.text()
        if self.y_centroid_offset_widget is not None:
            self.config['PID']['y centroid offset'] = self.y_centroid_offset_widget.text()
        if hasattr(self, 'log_toggle_widget'):
            self.config['EXECUTION']['log'] = self.log_toggle_widget.currentText()
        if hasattr(self, 'logdir_widget'):
            self.config['EXECUTION']['logdir'] = self.logdir_widget.text()

    def _validate_file_path(self, text_field):
        """Highlight file path text fields orange if the file is missing."""
        path = text_field.text().strip()
        if path and not os.path.isfile(path):
            text_field.setStyleSheet("background-color: #FFE0B2;")
        else:
            text_field.setStyleSheet("")

    def get_widget_value(self, widget):
        """Get the value from a widget"""
        if isinstance(widget, QLineEdit):
            return widget.text()
        elif isinstance(widget, QComboBox):
            return widget.currentText()
        elif hasattr(widget, 'text_field') and isinstance(widget.text_field, QLineEdit):
            return widget.text_field.text()
        else:
            return str(widget.text())

    def set_widget_value(self, widget, value):
        """Set the value of an input widget"""
        if isinstance(widget, QLineEdit):
            widget.setText(str(value))
        elif isinstance(widget, QComboBox):
            index = widget.findText(str(value))
            if index >= 0:
                widget.setCurrentIndex(index)
            else:
                widget.setCurrentText(str(value))
        elif hasattr(widget, 'text_field') and isinstance(widget.text_field, QLineEdit):
            widget.text_field.setText(str(value))

    def update_gui_from_config(self):
        """Update GUI widgets from the configuration"""
        if self.config_layout.count() == 0:
            self.create_widgets()
            return

        for i in range(self.config_layout.count()):
            section_frame = self.config_layout.itemAt(i).widget()
            if isinstance(section_frame, QFrame):
                section_layout = section_frame.layout()
                section_label = section_layout.itemAt(0).widget()
                section = section_label.text()

                self.update_section_widgets(section_layout, self.config[section])

        # Custom inline widgets not found by the generic walker
        if self.x_setpoint_widget is not None:
            self.x_setpoint_widget.setText(str(self.config['EXECUTION']['x setpoint']))
        if self.y_setpoint_widget is not None:
            self.y_setpoint_widget.setText(str(self.config['EXECUTION']['y setpoint']))
        if self.x_centroid_offset_widget is not None:
            self.x_centroid_offset_widget.setText(str(self.config['PID']['x centroid offset']))
        if self.y_centroid_offset_widget is not None:
            self.y_centroid_offset_widget.setText(str(self.config['PID']['y centroid offset']))
        if hasattr(self, 'log_toggle_widget'):
            self.log_toggle_widget.setCurrentText(str(self.config['EXECUTION']['log']))
        if hasattr(self, 'logdir_widget'):
            self.logdir_widget.setText(str(self.config['EXECUTION']['logdir']))

    def update_section_widgets(self, section_layout, config_section):
        """Update widgets in a section with values from the config"""
        for i in range(1, section_layout.count()):
            item = section_layout.itemAt(i).widget()
            if isinstance(item, QWidget):
                if isinstance(item, CollapsibleBox):
                    try:
                        content_widget = item.content_area.widget()
                        if content_widget:
                            content_layout = content_widget.layout()
                            if content_layout:
                                for j in range(content_layout.rowCount()):
                                    label_item = content_layout.itemAtPosition(j, 0)
                                    input_item = content_layout.itemAtPosition(j, 1)
                                    if label_item and input_item:
                                        label = label_item.widget()
                                        input_widget = input_item.widget()
                                        key = label.text()
                                        if key in config_section:
                                            self.set_widget_value(input_widget, config_section[key])
                            else:
                                print(f"Warning: CollapsibleBox content widget has no layout")
                        else:
                            print(f"Warning: CollapsibleBox content area has no widget")
                    except Exception as e:
                        print(f"Error updating CollapsibleBox: {e}")
                else:
                    item_layout = item.layout()
                    if item_layout and item_layout.count() == 2:
                        label = item_layout.itemAt(0).widget()
                        input_widget = item_layout.itemAt(1).widget()
                        key = label.text()
                        if key in config_section:
                            self.set_widget_value(input_widget, config_section[key])

    def run_config(self):
        """Run the qacits tracking algorithm with the current configuration"""
        self.update_config_from_gui()

        print("Current configuration:")
        for section in self.config.sections:
            print(f"[{section}]")
            for key, value in self.config[section].items():
                print(f"    {key} = {value}")
            print()

        # Close any existing plotter before creating a new one
        if hasattr(self, 'plotter') and self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:
                pass

        # Create plotter in the MAIN thread
        if self.config['EXECUTION'].get('plot', 'True').lower() == 'true':
            self.plotter = QacitsPlotter(figsize=(300, 450))
        else:
            self.plotter = None

        # Launch algorithm thread
        self.algorithm_thread = AlgorithmThread(
            camera=self.camera,
            aosystem=self.aosystem,
            config=self.config,
            spec_file=self.spec_file,
            my_event=self.my_event,
            main_window=self,
            param_update_queue=self.param_update_queue,
            centroid_offset_queue=self.centroid_offset_queue,
            centroid_offset_feedback=self.centroid_offset_feedback,
            warning_queue=self.warning_queue
        )
        
        # Connect signals
        self.algorithm_thread.plot_update_signal.connect(self.handle_plot_update)
        self.algorithm_thread.error_signal.connect(self.handle_thread_error)
        
        self.algorithm_thread.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Global exception hook: show all uncaught errors as popups
    def _exception_hook(exc_type, exc_value, exc_tb):
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_tb)
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(None, exc_type.__name__, str(exc_value))

    sys.excepthook = _exception_hook

    ex = QacitsConfigGUI()
    ex.show()
    sys.exit(app.exec_())