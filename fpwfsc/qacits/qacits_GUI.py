import sys
import os
import queue  # For thread-safe setpoint updates

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QComboBox, QPushButton,
                             QScrollArea, QFrame, QToolButton, QSizePolicy,
                             QFileDialog, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QParallelAnimationGroup, QPropertyAnimation, QAbstractAnimation, QTimer, QThread
from PyQt5.QtGui import QFont

import matplotlib.pyplot as plt
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
    
    def __init__(self, camera, aosystem, config, spec_file, my_event, main_window, setpoint_queue):
        super().__init__()
        self.camera = camera
        self.aosystem = aosystem
        self.config = config
        self.spec_file = spec_file
        self.my_event = my_event
        self.main_window = main_window  # Reference to main window, not plotter directly
        self.setpoint_queue = setpoint_queue  # Queue for setpoint updates

    def run(self):
        try:
            # Run the qacits tracking algorithm in this thread
            # Pass the signal and the queue
            original_run(camera=self.camera,
                         aosystem=self.aosystem,
                         config=self.config,
                         configspec=self.spec_file,
                         my_event=self.my_event,
                         plotter=None,  # Don't pass plotter directly
                         plot_signal=self.plot_update_signal if self.main_window.plotter else None,
                         setpoint_queue=self.setpoint_queue)  # Pass the queue
        except Exception as e:
            import traceback
            print(f"Error in algorithm thread: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
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
        
        # Queue for thread-safe setpoint updates
        self.setpoint_queue = queue.Queue()
        self.current_setpoint = None  # Will be set from config when loop starts
        
        # References to setpoint input widgets (set during create_widgets)
        self.x_setpoint_widget = None
        self.y_setpoint_widget = None

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
            try:
                self.algorithm_thread.terminate()
            except Exception:
                pass

        event.accept()
        os._exit(0)

    def initUI(self):
        self.setWindowTitle('miniQACITS')
        self.resize(325, 530)

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

        # Update Setpoint button
        self.update_setpoint_button = QPushButton('Update Setpoint')
        self.update_setpoint_button.setFont(QFont('Arial', 9))
        self.update_setpoint_button.clicked.connect(self.on_update_setpoint_clicked)
        self.update_setpoint_button.setEnabled(False)  # Disabled until loop starts
        self.update_setpoint_button.setStyleSheet("background-color: #4CAF50; color: white;")
        button_layout.addWidget(self.update_setpoint_button)

        self.run_stop_button = QPushButton('Run')
        self.run_stop_button.setFont(QFont('Arial', 10, QFont.Bold))
        self.run_stop_button.clicked.connect(self.toggle_run_stop)
        self.run_stop_button.setStyleSheet("background-color: green; color: white;")
        button_layout.addWidget(self.run_stop_button)

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

    def on_update_setpoint_clicked(self):
        """Handle update setpoint button click - reads from GUI text fields"""
        # Check if widgets are available
        if self.x_setpoint_widget is None or self.y_setpoint_widget is None:
            print("Error: Setpoint widgets not found. Cannot update setpoint.")
            return
        
        try:
            # Get values from the GUI text fields
            x_value = self.get_widget_value(self.x_setpoint_widget)
            y_value = self.get_widget_value(self.y_setpoint_widget)
            
            # Convert to float
            new_x = float(x_value)
            new_y = float(y_value)
            
            # Send to control loop via queue (note: y, x order to match numpy array convention)
            self.setpoint_queue.put((new_y, new_x))
            
            # Update stored value
            self.current_setpoint = (new_x, new_y)
            
            print(f"✓ Setpoint update queued from GUI: X={new_x:.2f}, Y={new_y:.2f}")
            
        except (ValueError, TypeError) as e:
            print(f"Error: Invalid setpoint values in GUI fields. Please check x setpoint and y setpoint are valid numbers.")
            print(f"Details: {e}")


    def check_thread_status(self):
        """Check if algorithm thread is still running"""
        if hasattr(self, 'algorithm_thread'):
            if not self.algorithm_thread.isRunning() and self.is_running:
                self.is_running = False
                self.run_stop_button.setText('Run')
                self.run_stop_button.setStyleSheet("background-color: green; color: white;")
                self.update_setpoint_button.setEnabled(False)  # Disable when stopped
                self.cleanup_resources()

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
            self.update_setpoint_button.setEnabled(True)  # Enable when running
            print("Running")
            self.my_event = threading.Event()
            
            # Clear the queue and update current setpoint from config before starting
            while not self.setpoint_queue.empty():
                try:
                    self.setpoint_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Get initial setpoint from config
            self.update_config_from_gui()
            self.current_setpoint = (
                float(self.config['EXECUTION']['x setpoint']),
                float(self.config['EXECUTION']['y setpoint'])
            )
            
            self.run_config()
        else:
            self.is_running = False
            self.run_stop_button.setText('Run')
            self.run_stop_button.setStyleSheet("background-color: green; color: white;")
            self.run_stop_button.setEnabled(False)
            self.update_setpoint_button.setEnabled(False)  # Disable when stopping
            print("Stopping")

            if hasattr(self, '_stopping') and self._stopping:
                print("Already attempting to stop, please wait...")
                return

            self._stopping = True

            if hasattr(self, 'my_event'):
                print("Setting stop event")
                self.my_event.set()

                if hasattr(self, 'algorithm_thread') and self.algorithm_thread.isRunning():
                    QTimer.singleShot(500, self.check_if_force_terminate)
                else:
                    self._stopping = False
                    self.cleanup_resources()

    def check_if_force_terminate(self):
        """Check if we need to force terminate the thread"""
        if hasattr(self, 'algorithm_thread') and self.algorithm_thread.isRunning():
            print("Thread still running after timeout, attempting to terminate...")
            try:
                self.algorithm_thread.terminate()
                self.algorithm_thread.wait(500)

                if self.algorithm_thread.isRunning():
                    print("Thread could not be terminated. Disconnecting it.")
                    self.algorithm_thread = None
            except Exception as e:
                print(f"Error terminating thread: {e}")
            finally:
                self.cleanup_resources()
        else:
            self.cleanup_resources()

    def cleanup_resources(self):
        """Clean up any resources that need to be explicitly closed"""
        if hasattr(self, '_stopping'):
            self._stopping = False

        self.run_stop_button.setEnabled(True)

        if hasattr(self, 'plotter') and self.plotter is not None:
            try:
                plt.close('all')
                if hasattr(self.plotter, 'close'):
                    self.plotter.close()
                elif hasattr(self.plotter, 'cleanup'):
                    self.plotter.cleanup()
                print("Plotter closed")
            except Exception as e:
                print(f"Error closing plotter: {e}")
            self.plotter = None

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
                item_widget = self.create_item_widget(key, value, section)
                section_layout.addWidget(item_widget)

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
        
        # Store references to setpoint widgets for easy access
        if section == 'EXECUTION' and key == 'x setpoint':
            self.x_setpoint_widget = input_widget
        elif section == 'EXECUTION' and key == 'y setpoint':
            self.y_setpoint_widget = input_widget

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

            browse_button = QPushButton("...")
            browse_button.setFixedWidth(30)
            browse_button.setFixedHeight(20)

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

            browse_button = QPushButton("...")
            browse_button.setFixedWidth(30)
            browse_button.setFixedHeight(20)

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
                                key = item_layout.itemAt(0).widget().text()
                                value = self.get_widget_value(item_layout.itemAt(1).widget())
                                self.config[section][key] = value

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

        # Create plotter in the MAIN thread
        if self.config['EXECUTION'].get('plot', 'True').lower() == 'true':
            self.plotter = QacitsPlotter(figsize=(300, 300))
        else:
            self.plotter = None

        # Launch algorithm thread
        self.algorithm_thread = AlgorithmThread(
            camera=self.camera,
            aosystem=self.aosystem,
            config=self.config,
            spec_file=self.spec_file,
            my_event=self.my_event,
            main_window=self,  # Pass reference to main window
            setpoint_queue=self.setpoint_queue  # Pass the queue
        )
        
        # Connect the signal to the slot
        self.algorithm_thread.plot_update_signal.connect(self.handle_plot_update)
        
        self.algorithm_thread.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = QacitsConfigGUI()
    ex.show()
    sys.exit(app.exec_())