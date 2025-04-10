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

from fpwfsc.fnf import gui_helper as helper
from fpwfsc.fnf.run import run
from fpwfsc.common import plotting_funcs as pf

def get_instrument_values():
    return {'MODELLING':{'wavelength (m)':'3e-6',
            'aperture':'NIRC2_large_hexagonal_mask'}}

class AlgorithmThread(QThread):
    update_plot = pyqtSignal(dict)

    def __init__(self, camera, aosystem, config, spec_file, my_event, plotter):
        super().__init__()
        self.camera = camera
        self.aosystem = aosystem
        self.config = config
        self.spec_file = spec_file
        self.my_event = my_event
        self.plotter = plotter

    def run(self):
        run(camera=self.camera,
            aosystem=self.aosystem,
            config=self.config,
            configspec=self.spec_file,
            my_event=self.my_event,
            plotter=self.plotter)

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.content_area = QScrollArea()
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setWidgetResizable(True)
        self.content_area.setFrameShape(QFrame.NoFrame)

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

    @pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if not checked else Qt.RightArrow)
        self.toggle_animation.setDirection(QAbstractAnimation.Forward if not checked else QAbstractAnimation.Backward)
        self.toggle_animation.start()

    def setContentLayout(self, layout):
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


class ConfigEditorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.expert_options = {}
        self.config_file = 'fpwfsc/fnf/FF_software_sim.ini'
        self.spec_file = 'fpwfsc/fnf/FF_software.spec'
        self.is_running = False
        self.my_event = threading.Event()

        self.initUI()

    def initUI(self):
        # Set up the main window
        self.setWindowTitle('Fast and Furious')
        self.resize(425, 875)

        main_layout = QVBoxLayout(self)
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
        scroll_content = QWidget(scroll)

        self.layout = QVBoxLayout(scroll_content)
        self.layout.setSpacing(2)  # Reduce from default
        scroll.setWidget(scroll_content)

        # Load configuration and create widgets. creates self.config
        self.load_config(initial_load=True)
        self.load_expert_options()
        self.create_widgets()
        # Set up the button layout
        button_layout = QGridLayout()
        button_layout.setVerticalSpacing(2)
        button_layout.setHorizontalSpacing(5)

        # Create and add buttons
        self.run_stop_button = QPushButton('Run')
        self.run_stop_button.setFont(QFont('Arial', 10, QFont.Bold))
        self.run_stop_button.clicked.connect(self.toggle_run_stop)
        self.run_stop_button.setStyleSheet("background-color: green; color: white;")
        button_layout.addWidget(self.run_stop_button, 0, 0)

        save_centroids_button = QPushButton('Save centroids')
        save_centroids_button.clicked.connect(self.save_centroids)
        button_layout.addWidget(save_centroids_button, 0, 1)

        save_button = QPushButton('Save configuration')
        save_button.clicked.connect(self.save_config)
        button_layout.addWidget(save_button, 1, 0)

        load_centroids_button = QPushButton('Load centroids')
        load_centroids_button.clicked.connect(self.load_centroids)
        button_layout.addWidget(load_centroids_button, 1, 1)

        load_config_button = QPushButton('Load configuration')
        load_config_button.clicked.connect(lambda: self.load_config(None))
        button_layout.addWidget(load_config_button, 2, 0)

        load_instrument_button = QPushButton('Load values from instrument')
        load_instrument_button.clicked.connect(self.load_instrument_values)
        button_layout.addWidget(load_instrument_button, 2, 1)

        main_layout.addLayout(button_layout)

        # Add timer to check thread status
        self.thread_check_timer = QTimer()
        self.thread_check_timer.timeout.connect(self.check_thread_status)
        self.thread_check_timer.start(1000)  # Check every 1000ms

        self.default_hardware = helper.valid_instruments[0]  # Default initial value
        self.on_hardware_changed(self.default_hardware)

    def on_hardware_changed(self, selected_hardware):
        try:
            print(f"Loading {selected_hardware}...")
            self.camera, self.aosystem = helper.load_instruments(selected_hardware,
                                                                 camargs={},
                                                                 aoargs={'rotation_angle_dm':
                                                                         self.config['MODELLING']['rotation angle dm (deg)']})
            print(f"{selected_hardware} loaded successfully")

        except Exception as e:
            print(f"Error loading {selected_hardware}: {str(e)}")
            print("Loading Sim as default")
            # Revert only the hardware selector to previous working selection
            default_index = self.hardware_select.findText('Sim')
            if default_index >= 0:
                self.hardware_select.blockSignals(True)  # Prevent triggering another callback
                self.hardware_select.setCurrentIndex(default_index)
                self.hardware_select.blockSignals(False)

    def check_thread_status(self):
        if hasattr(self, 'algorithm_thread'):
            if not self.algorithm_thread.isRunning() and self.is_running:
                self.is_running = False
                self.run_stop_button.setText('Run')
                self.run_stop_button.setStyleSheet("background-color: green; color: white;")

    def load_instrument_values(self):
        print("Loading values from instrument")
        instrument_values = get_instrument_values()

        for section, values in instrument_values.items():
            if section in self.config:
                for key, value in values.items():
                    if key in self.config[section]:
                        if self.validate_config_value(section, key, value):
                            self.config[section][key] = value
                        else:
                            print(f"Invalid value for {section}.{key}: {value}")
        self.update_gui_from_config()

    def validate_config_value(self, section, key, value):
        validator = Validator()
        test_config = ConfigObj(configspec=self.spec_file)
        test_config[section] = {}
        test_config[section][key] = value

        result = test_config.validate(validator, preserve_errors=True)
        return result[section][key] is True

    def load_expert_options(self):
        # Load the expert options from a separate file
        try:
            expert_config = ConfigObj('expert_options.ini')
            for section, options in expert_config.items():
                if 'expert_options' in options:
                    if isinstance(options['expert_options'], list):
                        self.expert_options[section] = [opt.strip() for opt in options['expert_options']]
                    elif isinstance(options['expert_options'], str):
                        self.expert_options[section] = [opt.strip() for opt in options['expert_options'].split(',')]
                    else:
                        print(f"Unexpected type for expert_options in section {section}")
        except IOError as e:
            print(f"Error loading expert options: {e}")
            # If the file doesn't exist, we'll just have an empty expert_options dict
            pass

    def toggle_run_stop(self):
        #This controls the toggle run button.
        if not self.is_running:
            self.is_running = True
            self.run_stop_button.setText('Stop')
            self.run_stop_button.setStyleSheet("background-color: red; color: white;")
            print("Running")
            #Create fresh event flag and thread
            self.my_event = threading.Event()
            self.run_config()  # Call run_config when starting
        else:
            #Stop it.
            self.is_running = False
            self.run_stop_button.setText('Run')
            self.run_stop_button.setStyleSheet("background-color: green; color: white;")
            print("Stopping")
            if hasattr(self, 'my_event'):
                self.my_event.set()
                self.algorithm_thread.wait()

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
        except (IOError, ConfigObjError) as e:
            print(f"Error loading configuration: {e}")
            if initial_load:
                print("Error loading default configuration. Exiting.")
                sys.exit(1)

    def create_widgets(self):
        # Create widgets for each configuration section
        for section, items in self.config.items():
            section_frame = QFrame()
            section_frame.setFrameShape(QFrame.StyledPanel)
            section_layout = QVBoxLayout(section_frame)
            section_layout.setSpacing(4)

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

            # Add expert options in a collapsible box
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

            self.layout.addWidget(section_frame)

    def create_item_widget(self, key, value, section):
        # Create a widget for a single configuration item
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(key)
        layout.addWidget(label)

        input_widget = self.create_input_widget(section, key, value)
        layout.addWidget(input_widget)

        # Set tooltips for both label and input widget
        tooltip = helper.get_help_message(section, key)
        label.setToolTip(tooltip)
        input_widget.setToolTip(tooltip)

        return widget

    def create_input_widget(self, section, key, value):
        # Create an appropriate input widget based on the configuration specification
        spec = self.get_spec_for_key(f"{section}.{key}")

        if spec:
            if 'option(' in spec:
                input_widget = QComboBox()
                options = spec.split('option(')[1].split(')')[0].replace("'", "").split(',')
                input_widget.addItems([opt.strip() for opt in options])
                input_widget.setCurrentText(str(value).strip())
            elif 'boolean' in spec:
                input_widget = QComboBox()
                input_widget.addItems(['True', 'False'])
                input_widget.setCurrentText(str(value))
            else:
                input_widget = QLineEdit(str(value))
        else:
            input_widget = QLineEdit(str(value))

        #input_widget.setMinimumHeight(20)  # Ensure minimum height for all input widgets
        input_widget.setFixedHeight(20)  # Ensure minimum 20height for all input widgets
        return input_widget

    def get_spec_for_key(self, key):
        # Get the specification for a given key from the config spec
        keys = key.split('.')
        spec = self.config.configspec
        for k in keys:
            if k in spec:
                spec = spec[k]
            else:
                return None
        return spec

    def save_config(self):
        # Save the current configuration to a file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        default_name = "config.ini"
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
        self.config = ConfigObj(self.config, configspec=self.spec_file)
        validator = Validator()
        self.config.validate(validator, preserve_errors=True)

    def save_section_widgets(self, section_layout, config_section):
        # Save the values from widgets in a section to the config
        for i in range(1, section_layout.count()):  # Start from 1 to skip the section label
            item = section_layout.itemAt(i).widget()
            if isinstance(item, QWidget):
                if isinstance(item, CollapsibleBox):
                    # Handle expert options
                    content_layout = item.content_area.layout()
                    for j in range(content_layout.rowCount()):
                        label_item = content_layout.itemAtPosition(j, 0)
                        input_item = content_layout.itemAtPosition(j, 1)
                        if label_item and input_item:
                            label = label_item.widget()
                            input_widget = input_item.widget()
                            key = label.text()
                            value = self.get_widget_value(input_widget)
                            if key in config_section:
                                config_section[key] = value
                else:
                    # Handle regular options
                    item_layout = item.layout()
                    if item_layout and item_layout.count() == 2:
                        label = item_layout.itemAt(0).widget()
                        input_widget = item_layout.itemAt(1).widget()
                        key = label.text()
                        value = self.get_widget_value(input_widget)
                        if key in config_section:
                            config_section[key] = value

    def get_widget_value(self, widget):
        if isinstance(widget, QLineEdit):
            return widget.text()
        elif isinstance(widget, QComboBox):
            return widget.currentText()
        else:
            return str(widget.text())  # Fallback for any other widget types

    def save_centroids(self):
        # Placeholder function for saving centroids
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Centroids", "", "All Files (*)")
        if file_name:
            print("Centroids saved")

    def load_centroids(self):
        # Placeholder function for loading centroids
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Centroids", "", "All Files (*)")
        if file_name:
            print("XYZ loaded")

    def run_config(self):

        # Update the configuration from the GUI and print it
        self.update_config_from_gui()
        validator = Validator()
        results = self.config.validate(validator, preserve_errors=True)
        if results is not True:
            # Format validation errors into readable messages
            error_messages = []
            for section_list, key, error in flatten_errors(self.config, results):
                if key is not None:
                    section_str = '.'.join(section_list)
                    error_messages.append(f"Invalid value in [{section_str}] {key}: {str(error)}")
                    # Print validation errors
            print("\nConfiguration Validation Failed:")
            for msg in error_messages:
                print(f"  {msg}")
            # Reset run/stop button state
            self.is_running = False
            self.run_stop_button.setText('Run')
            self.run_stop_button.setStyleSheet("background-color: green; color: white;")
            return

        else:
            # Close any existing plot windows
            plt.close('all')
            print("Current configuration:")
            for section in self.config.sections:
                print(f"[{section}]")
                for key, value in self.config[section].items():
                    print(f"    {key} = {value}")
                print()

            if self.config['LOOP_SETTINGS']['Plot']:
                self.plotter = pf.LivePlotter()
            else:
                self.plotter = None
            self.my_event = threading.Event()

            print("Hardware")
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

    def update_gui_from_config(self):
        # Update the GUI widgets with values from the loaded configuration
        for i in range(self.layout.count()):
            section_frame = self.layout.itemAt(i).widget()
            if isinstance(section_frame, QFrame):
                section_layout = section_frame.layout()
                section_label = section_layout.itemAt(0).widget()
                section = section_label.text().strip('<b>').strip('</b>')

                self.update_section_widgets(section_layout, self.config[section])

    def update_section_widgets(self, section_layout, config_section):
        # Update widgets in a section with values from the config
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

    def set_widget_value(self, widget, value):
        # Set the value of an input widget
        if isinstance(widget, QLineEdit):
            widget.setText(str(value))
        elif isinstance(widget, QComboBox):
            index = widget.findText(str(value))
            if index >= 0:
                widget.setCurrentIndex(index)
            else:
                widget.setCurrentText(str(value))

# Main execution block
if __name__ == '__main__':
    # Create the application
    app = QApplication(sys.argv)

    # Create and show the main window
    ex = ConfigEditorGUI()
    ex.show()

    # Start the event loop
    sys.exit(app.exec_())
