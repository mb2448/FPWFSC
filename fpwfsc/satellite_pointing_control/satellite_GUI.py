import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QComboBox, QPushButton,
                             QScrollArea, QFrame, QToolButton, QSizePolicy,
                             QFileDialog, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QParallelAnimationGroup, QPropertyAnimation, QAbstractAnimation, QTimer, QThread
from PyQt5.QtGui import QFont

import matplotlib.pyplot as plt
from configobj import ConfigObj, ConfigObjError, flatten_errors
from validate import Validator
import threading
import time
from pathlib import Path

# Import the satellite GUI helper module
import satellite_gui_helper as helper

# Import the satellite plotter
from satellite_plotter_qt import LiveSquarePlotter

# Import the run function from run_satellite.py
from run_satellite import run as original_run
from run_satellite import apply_waffle, remove_waffle

class AlgorithmThread(QThread):
    """Thread to run the satellite tracking algorithm with proper stop handling"""
    def __init__(self, camera, aosystem, config, spec_file, my_event, plotter):
        super().__init__()
        self.camera = camera
        self.aosystem = aosystem
        self.config = config
        self.spec_file = spec_file
        self.my_event = my_event
        self.plotter = plotter
        self.original_update = None
        
        # Store the original plotter update function if it exists
        if self.plotter is not None and hasattr(self.plotter, 'update'):
            self.original_update = self.plotter.update
            
            # Create a wrapped update function that checks for stop event
            def wrapped_update(*args, **kwargs):
                if self.my_event.is_set():
                    # Don't update if we're stopping
                    return
                return self.original_update(*args, **kwargs)
            
            # Replace the update function
            self.plotter.update = wrapped_update

    def run(self):
        try:
            # Run the satellite tracking algorithm in this thread
            # The algorithm will check the my_event periodically
            original_run(camera=self.camera,
                         aosystem=self.aosystem,
                         config=self.config,
                         configspec=self.spec_file,
                         my_event=self.my_event,
                         plotter=self.plotter)
        except Exception as e:
            print(f"Error in algorithm thread: {str(e)}")
        finally:
            # Make sure we restore the original update function
            if self.plotter is not None and self.original_update is not None:
                self.plotter.update = self.original_update
            print("Algorithm thread finished")

class CollapsibleBox(QWidget):
    """A collapsible box widget for expert options"""
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.clicked.connect(self.on_clicked)  # Use clicked instead of pressed

        self.content_area = QScrollArea()
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)  # Allow content to expand
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setWidgetResizable(True)
        self.content_area.setFrameShape(QFrame.NoFrame)
        self.content_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show scrollbar when needed

        # Create a widget to hold the content
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
        
        # Connect the animation finished signal
        self.toggle_animation.finished.connect(self.animation_finished)
        
        # Track animation state
        self.is_animating = False
        self.is_expanded = False

    @pyqtSlot()
    def on_clicked(self):
        """Handle toggle button click"""
        if self.is_animating:
            # Ignore clicks during animation
            return
            
        self.is_animating = True
        self.toggle_button.setEnabled(False)  # Disable button during animation
        
        # Set arrow direction
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
        self.toggle_button.setEnabled(True)  # Re-enable button
        self.is_expanded = not self.is_expanded  # Toggle state

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


class SatelliteConfigGUI(QWidget):
    """Main GUI class for the satellite configuration editor"""
    def __init__(self):
        super().__init__()
        
        # Get file paths
        script_dir = Path(__file__).parent
        config_path = script_dir/"satellite_config.ini"
        spec_path = script_dir/"satellite_config.spec"

        self.config_file = str(config_path)
        self.spec_file = str(spec_path)
        self.is_running = False
        self.my_event = threading.Event()
        self.plotter = None
        self._stopping = False  # Flag to track stopping state

        self.initUI()
        
    def closeEvent(self, event):
        """Handle window close event - ensure all resources are cleaned up"""
        print("Window closing, cleaning up resources...")
        
        # If we're still running, try to stop gracefully
        if self.is_running:
            self.is_running = False
            if hasattr(self, 'my_event'):
                self.my_event.set()
        
        # Force terminate any running thread
        if hasattr(self, 'algorithm_thread') and self.algorithm_thread.isRunning():
            try:
                print("Terminating algorithm thread...")
                self.algorithm_thread.terminate()
                self.algorithm_thread.wait(500)  # Wait half a second max
            except Exception as e:
                print(f"Error terminating thread: {e}")
        
        # Clean up resources
        self.cleanup_resources()
        event.accept()

    def initUI(self):
        # Set up the main window
        self.setWindowTitle('Satellite Tracking')
        self.resize(325, 500)  # Reduced height

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(4)  # Reduced overall spacing
        main_layout.setContentsMargins(10, 10, 10, 10)  # Reduced margins
        
        # Add hardware selection widget at the top
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

        # Create a scrollable area for the config options
        scroll = QScrollArea(self)
        main_layout.addWidget(scroll)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show scrollbar when needed
        scroll_content = QWidget(scroll)

        self.layout = QVBoxLayout(scroll_content)
        self.layout.setSpacing(2)  # Reduce spacing between sections
        self.layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        scroll.setWidget(scroll_content)

        # Load configuration and create widgets
        self.load_config(initial_load=True)
        self.create_widgets()
        
        # Set up the button layout with more vertical space before it
        main_layout.addSpacing(10)  # Add space before buttons

        # Set up the button layout
        button_layout = QVBoxLayout()  # Change to vertical layout
        button_layout.setSpacing(10)  # Spacing between button sections

        # Create a horizontal layout for Waffle control buttons with different shading
        waffle_buttons_layout = QHBoxLayout()
        waffle_buttons_layout.setSpacing(10)  # Horizontal spacing between buttons

        # Add Waffle and Remove Waffle buttons with different shading
        add_waffle_button = QPushButton('Add Waffle')
        add_waffle_button.clicked.connect(self.on_add_waffle)
        add_waffle_button.setStyleSheet("background-color: #E8E8FF;")  # Light blue shade
        waffle_buttons_layout.addWidget(add_waffle_button)

        remove_waffle_button = QPushButton('Remove Waffle')
        remove_waffle_button.clicked.connect(self.on_remove_waffle)
        remove_waffle_button.setStyleSheet("background-color: #FFE8E8;")  # Light red shade
        waffle_buttons_layout.addWidget(remove_waffle_button)

        # Add the waffle buttons layout to the main button layout
        button_layout.addLayout(waffle_buttons_layout)

        # Add spacing between button rows
        button_layout.addSpacing(5)

        # Create a horizontal layout for Save and Load buttons
        config_buttons_layout = QHBoxLayout()
        config_buttons_layout.setSpacing(10)  # Horizontal spacing between buttons

        # Add Save and Load buttons
        save_button = QPushButton('Save configuration')
        save_button.clicked.connect(self.save_config)
        config_buttons_layout.addWidget(save_button)

        load_config_button = QPushButton('Load configuration')
        load_config_button.clicked.connect(lambda: self.load_config(None))
        config_buttons_layout.addWidget(load_config_button)

        # Add this horizontal layout to the main button layout
        button_layout.addLayout(config_buttons_layout)

        # Add spacing between button rows
        button_layout.addSpacing(5)

        # Add Run button at the bottom
        self.run_stop_button = QPushButton('Run')
        self.run_stop_button.setFont(QFont('Arial', 10, QFont.Bold))
        self.run_stop_button.clicked.connect(self.toggle_run_stop)
        self.run_stop_button.setStyleSheet("background-color: green; color: white;")
        button_layout.addWidget(self.run_stop_button)  # Run button at the bottom

        # Add the entire button layout to the main layout
        main_layout.addLayout(button_layout)

        # Add timer to check thread status
        self.thread_check_timer = QTimer()
        self.thread_check_timer.timeout.connect(self.check_thread_status)
        self.thread_check_timer.start(1000)  # Check every 1000ms

        # Initialize with default hardware
        self.default_hardware = helper.valid_instruments[0]
        self.on_hardware_changed(self.default_hardware)

    def on_hardware_changed(self, selected_hardware):
        """Handle changes to the hardware selection dropdown"""
        try:
            print(f"Loading {selected_hardware}...")
            # Use helper function to load instruments
            self.camera, self.aosystem = helper.load_instruments(selected_hardware)
            print(f"{selected_hardware} loaded successfully")

        except Exception as e:
            print(f"Error loading {selected_hardware}: {str(e)}")
            print("Loading Sim as default")
            # Revert to Sim if there's an error
            default_index = self.hardware_select.findText('Sim')
            if default_index >= 0:
                self.hardware_select.blockSignals(True)  # Prevent triggering another callback
                self.hardware_select.setCurrentIndex(default_index)
                self.hardware_select.blockSignals(False)
                self.camera, self.aosystem = helper.load_instruments('Sim')

    def check_thread_status(self):
        """Check if algorithm thread is still running"""
        if hasattr(self, 'algorithm_thread'):
            if not self.algorithm_thread.isRunning() and self.is_running:
                self.is_running = False
                self.run_stop_button.setText('Run')
                self.run_stop_button.setStyleSheet("background-color: green; color: white;")
                self.cleanup_resources()

    def toggle_run_stop(self):
        """Handle Run/Stop button press"""
        if not self.is_running:
            self.is_running = True
            self.run_stop_button.setText('Stop')
            self.run_stop_button.setStyleSheet("background-color: red; color: white;")
            print("Running")
            # Create fresh event flag and thread
            self.my_event = threading.Event()
            self.run_config()  # Call run_config when starting
        else:
            # Stop it
            self.is_running = False
            self.run_stop_button.setText('Run')
            self.run_stop_button.setStyleSheet("background-color: green; color: white;")
            # Temporarily disable the button while stopping
            self.run_stop_button.setEnabled(False)
            print("Stopping")
            
            # Make sure we have only one stopping attempt at a time
            if hasattr(self, '_stopping') and self._stopping:
                print("Already attempting to stop, please wait...")
                return
                
            self._stopping = True
                
            if hasattr(self, 'my_event'):
                print("Setting stop event")
                self.my_event.set()
                
                # Force termination of thread if it doesn't respond to the event
                if hasattr(self, 'algorithm_thread') and self.algorithm_thread.isRunning():
                    # Give the thread a shorter time to respond to the event (500ms)
                    QTimer.singleShot(500, self.check_if_force_terminate)
                else:
                    self._stopping = False
                    self.cleanup_resources()
    
    def on_add_waffle(self):
        """
        Handler for the Add Waffle button.
        This will apply a waffle pattern to the DM using the amplitude from the config.
        """
        # First update config from GUI to ensure we have the latest values
        self.update_config_from_gui()
        
        # Get the waffle amplitude from the config
        try:
            # Get the waffle amplitude from the AO section
            waffle_amplitude = float(self.config['AO']['waffle mode amplitude'])
            
            # Apply the waffle pattern (function already imported at top of file)
            success = apply_waffle(aosystem=self.aosystem, waffle_amplitude=waffle_amplitude)
            
            if success:
                print(f"Successfully applied waffle pattern with amplitude {waffle_amplitude}")
            else:
                print("Failed to apply waffle pattern")
                
        except KeyError:
            print("Error: 'waffle mode amplitude' not found in config")
        except ValueError as e:
            print(f"Error parsing waffle amplitude: {e}")
        except Exception as e:
            print(f"Unexpected error applying waffle: {e}")
    
    def on_remove_waffle(self):
        """
        Handler for the Remove Waffle button.
        This will remove the waffle pattern from the DM.
        """
        try:
            # Remove the waffle pattern (function already imported at top of file)
            success = remove_waffle(aosystem=self.aosystem)
            
            if success:
                print("Successfully removed waffle pattern")
            else:
                print("Failed to remove waffle pattern")
                
        except Exception as e:
            print(f"Unexpected error removing waffle: {e}")

    def check_if_force_terminate(self):
        """Check if we need to force terminate the thread"""
        if hasattr(self, 'algorithm_thread') and self.algorithm_thread.isRunning():
            print("Thread still running after timeout, attempting to terminate...")
            try:
                # Try to terminate the thread if it's still running
                self.algorithm_thread.terminate()
                self.algorithm_thread.wait(500)  # Wait a short time for termination
                
                # If still running after terminate, disconnect it and let it go
                if self.algorithm_thread.isRunning():
                    print("Thread could not be terminated. Disconnecting it.")
                    self.algorithm_thread = None  # Disconnect the thread reference
            except Exception as e:
                print(f"Error terminating thread: {e}")
            finally:
                # Clean up resources regardless of thread state
                self.cleanup_resources()
        else:
            self.cleanup_resources()
    
    def check_thread_finished(self):
        """Check if the thread has finished, and if not, wait for it"""
        if hasattr(self, 'algorithm_thread') and self.algorithm_thread.isRunning():
            print("Waiting for algorithm thread to finish...")
            # Wait with a timeout
            self.algorithm_thread.wait(1000)  # Wait up to 1 second
            
            # If still running, use a non-blocking approach
            if self.algorithm_thread.isRunning():
                print("Thread still running, will check again...")
                QTimer.singleShot(100, self.check_thread_finished)
            else:
                print("Thread finished")
                self.cleanup_resources()
        else:
            print("Thread not running")
            self.cleanup_resources()
    
    def cleanup_resources(self):
        """Clean up any resources that need to be explicitly closed"""
        # Reset the stopping flag if it exists
        if hasattr(self, '_stopping'):
            self._stopping = False
            
        # Enable the Run button immediately
        self.run_stop_button.setEnabled(True)
            
        # Close the plotter window if it exists
        if hasattr(self, 'plotter') and self.plotter is not None:
            try:
                # Try to close the plot window
                plt.close('all')  # Close all matplotlib windows
                
                # If the plotter has a close or cleanup method, call it
                if hasattr(self.plotter, 'close'):
                    self.plotter.close()
                elif hasattr(self.plotter, 'cleanup'):
                    self.plotter.cleanup()
                
                print("Plotter closed")
            except Exception as e:
                print(f"Error closing plotter: {e}")
            
            self.plotter = None

    def load_config(self, file_name=None, initial_load=False):
        """
        Load a configuration file and update the GUI.
        If initial_load is True, use the default config file.
        If file_name is None and not initial_load, open a file dialog to select a file.
        """
        if initial_load:
            file_name = self.config_file
        elif file_name is None:
            # Open file dialog
            dialog = QFileDialog(self)
            dialog.setNameFilter("INI Files (*.ini)")
            dialog.setFileMode(QFileDialog.ExistingFile)
            dialog.setViewMode(QFileDialog.List)

            if dialog.exec_():
                file_name = dialog.selectedFiles()[0]
            else:
                print("No file selected.")
                return  # User cancelled file selection

        try:
            new_config = ConfigObj(file_name, configspec=self.spec_file)
            validator = Validator()
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
        ####
        except (IOError, ConfigObjError) as e:
         print(f"Error loading configuration: {e}")
         if initial_load:
             # If there's an error loading the initial configuration, 
             # create a basic config based on the defaults in the spec file
             print("Error loading default configuration. Creating config from spec file.")
             try:
                 # Create a new config with the spec file
                 self.config = ConfigObj(configspec=self.spec_file)

                 # Apply defaults from the spec file
                 validator = Validator()
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
             except Exception as ex:
                 print(f"Error creating config from spec file: {ex}")
                 # Create a minimal fallback configuration if spec file processing fails
                 self.config = ConfigObj(configspec=self.spec_file)

                 # Add minimal required sections and values
                 self.config['HITCHHIKER MODE'] = {
                     'hitchhike': 'False',
                     'imagedir': '/'
                 }

                 self.config['EXECUTION'] = {
                     'plot': 'True',
                     'N iterations': '100',
                     'x setpoint': '512.0',
                     'y setpoint': '512.0',
                     'spot search radius (pix)': '60',
                     'radius tolerance (pix)': '20'
                 }

                 self.config['AO'] = {
                     'waffle mode amplitude': '150e-9',
                     'tip tilt gain': '-250e-9',
                     'tip tilt angle (deg)': '0',
                     'tip tilt flip x': 'False',
                     'tip tilt flip y': 'False'
                 }

                 self.config['PID'] = {
                     'Kp': '0.5',
                     'Ki': '0.1',
                     'Kd': '0.0',
                     'output_limits': '3'
                 }

                 # Validate the minimal config
                 validator = Validator()
                 self.config.validate(validator, preserve_errors=True)

                 print("Created minimal fallback configuration")

             self.update_gui_from_config() 
    
    
    ####
    def create_widgets(self):
        """Create widgets for each configuration section"""
        # Clear existing widgets
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        
        # Create widgets for each configuration section
        for section, items in self.config.items():
            section_frame = QFrame()
            section_frame.setFrameShape(QFrame.StyledPanel)
            section_layout = QVBoxLayout(section_frame)
            section_layout.setSpacing(2)  # Reduced spacing
            section_layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins

            section_label = QLabel(f"<b>{section}</b>")
            section_layout.addWidget(section_label)

            regular_options = []
            expert_options = []

            # Separate regular and expert options
            for key, value in items.items():
                if helper.is_expert_option(section, key):
                    expert_options.append((key, value))
                else:
                    regular_options.append((key, value))

            # Add regular options
            for key, value in regular_options:
                item_widget = self.create_item_widget(key, value, section)
                section_layout.addWidget(item_widget)

            # Add expert options in a collapsible box if any exist
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

                    # Set tooltips for both label and input widget
                    tooltip = helper.get_help_message(section, key)
                    label.setToolTip(tooltip)
                    input_widget.setToolTip(tooltip)

                expert_box.setContentLayout(expert_layout)
                section_layout.addWidget(expert_box)
                
            # Compact sections with limited options - no maximum height constraints
            if len(regular_options) <= 2 and len(expert_options) == 0:
                # Use minimum height for sections with few options
                section_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

            self.layout.addWidget(section_frame)

    def create_item_widget(self, key, value, section):
        """Create a widget for a single configuration item"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(key)
        label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)  # Make label use minimum space
        layout.addWidget(label)

        input_widget = self.create_input_widget(section, key, value)
        layout.addWidget(input_widget)

        # Set tooltips for both label and input widget
        tooltip = helper.get_help_message(section, key)
        label.setToolTip(tooltip)
        input_widget.setToolTip(tooltip)

        return widget
    
    def create_input_widget(self, section, key, value):
        """Create an appropriate input widget based on the configuration specification"""
        # Get the spec for this key
        spec = self.get_spec_for_key(f"{section}.{key}")

        if helper.is_directory_option(section, key):
            # Directory selection - create a widget with a text field and browse button
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            # Text field for the directory path
            text_field = QLineEdit(str(value))
            text_field.setFixedHeight(20)

            # Browse button
            browse_button = QPushButton("...")
            browse_button.setFixedWidth(30)
            browse_button.setFixedHeight(20)

            # Connect browse button to directory selection dialog
            def browse_directory():
                directory = QFileDialog.getExistingDirectory(
                    None, "Select Directory", text_field.text(),
                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
                )
                if directory:
                    text_field.setText(directory)

            browse_button.clicked.connect(browse_directory)

            # Add widgets to layout
            layout.addWidget(text_field, 1)  # Text field takes most of the space
            layout.addWidget(browse_button, 0)  # Browse button takes minimal space

            # Save reference to text field for getting/setting value
            widget.text_field = text_field

            return widget

        elif spec and 'boolean' in spec:
            # Boolean values get dropdown
            input_widget = QComboBox()
            input_widget.addItems(['True', 'False'])
            input_widget.setCurrentText(str(value))
            input_widget.setFixedHeight(20)
            return input_widget
        elif spec and 'option(' in spec:
            # Options get dropdown
            input_widget = QComboBox()
            options = spec.split('option(')[1].split(')')[0].replace("'", "").split(',')
            input_widget.addItems([opt.strip() for opt in options])
            input_widget.setCurrentText(str(value).strip())
            input_widget.setFixedHeight(20)
            return input_widget
        else:
            # Everything else gets a text field
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
        default_name = "satellite_config.ini"
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
        for i in range(self.layout.count()):
            section_frame = self.layout.itemAt(i).widget()
            if isinstance(section_frame, QFrame):
                section_layout = section_frame.layout()
                section_label = section_layout.itemAt(0).widget()
                section = section_label.text().strip('<b>').strip('</b>')

                for j in range(1, section_layout.count()):
                    item = section_layout.itemAt(j).widget()
                    if isinstance(item, QWidget):
                        if isinstance(item, CollapsibleBox):
                            # Handle expert options
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
                            # Handle regular options
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
            # This is our directory selection widget
            return widget.text_field.text()
        else:
            return str(widget.text())  # Fallback for any other widget types    

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
            # This is our directory selection widget
            widget.text_field.setText(str(value))

    def update_gui_from_config(self):
        """Update GUI widgets from the configuration"""
        # If no widgets exist yet, create them
        if self.layout.count() == 0:
            self.create_widgets()
            return
            
        # Update existing widgets
        for i in range(self.layout.count()):
            section_frame = self.layout.itemAt(i).widget()
            if isinstance(section_frame, QFrame):
                section_layout = section_frame.layout()
                section_label = section_layout.itemAt(0).widget()
                section = section_label.text().strip('<b>').strip('</b>')

                self.update_section_widgets(section_layout, self.config[section])

    def update_section_widgets(self, section_layout, config_section):
        """Update widgets in a section with values from the config"""
        for i in range(1, section_layout.count()):  # Start from 1 to skip the section label
            item = section_layout.itemAt(i).widget()
            if isinstance(item, QWidget):
                if isinstance(item, CollapsibleBox):
                    # Handle expert options
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
                    # Handle regular options
                    item_layout = item.layout()
                    if item_layout and item_layout.count() == 2:
                        label = item_layout.itemAt(0).widget()
                        input_widget = item_layout.itemAt(1).widget()
                        key = label.text()
                        if key in config_section:
                            self.set_widget_value(input_widget, config_section[key])

    def run_config(self):
        """Run the satellite tracking algorithm with the current configuration"""
        # Update the configuration from the GUI
        self.update_config_from_gui()
        
        # Print configuration for debugging
        print("Current configuration:")
        for section in self.config.sections:
            print(f"[{section}]")
            for key, value in self.config[section].items():
                print(f"    {key} = {value}")
            print()
        
        # Create plotter if enabled
        if self.config['EXECUTION'].get('plot', 'True').lower() == 'true':
            self.plotter = LiveSquarePlotter(figsize=(300, 600))
        else:
            self.plotter = None
            
        # Launch in thread using custom QThread class
        self.algorithm_thread = AlgorithmThread(
            camera=self.camera,
            aosystem=self.aosystem,
            config=self.config,
            spec_file=self.spec_file,
            my_event=self.my_event,
            plotter=self.plotter
        )
        self.algorithm_thread.start()


# Main execution block
if __name__ == '__main__':
    # Create the application
    app = QApplication(sys.argv)

    # Create and show the main window
    ex = SatelliteConfigGUI()
    ex.show()

    # Start the event loop
    sys.exit(app.exec_())