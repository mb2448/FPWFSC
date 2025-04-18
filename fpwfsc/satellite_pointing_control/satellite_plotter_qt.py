import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
from pyqtgraph import ColorMap

class LiveSquarePlotter(QtWidgets.QWidget):
    def __init__(self, initial_setpoint=None, figsize=(600, 600)):
        # Initialize the QApplication if it hasn't been created yet
        if not QtWidgets.QApplication.instance():
            self.app = QtWidgets.QApplication(sys.argv)
        else:
            self.app = QtWidgets.QApplication.instance()
        
        super().__init__()
        
        # Set up the window
        self.setWindowTitle("Live Square Plotter")
        self.resize(*figsize)
        
        # Create a layout
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        
        # Create the main image plot
        self.main_plot = pg.PlotWidget()
        layout.addWidget(self.main_plot)
        
        # Create the residuals plot
        self.residuals_plot = pg.PlotWidget()
        layout.addWidget(self.residuals_plot)
        
        # Create a label to display the residual error
        self.residual_label = QtWidgets.QLabel("Residual Error: N/A")
        self.residual_label.setFont(QtGui.QFont("Arial", 12))
        layout.addWidget(self.residual_label)
        
        # Initialize tracking variables
        self.centers_history = []
        self.setpoints_history = []
        if initial_setpoint is not None:
            self.setpoints_history.append(initial_setpoint)
        
        # Initialize plot elements
        self.img_item = pg.ImageItem()
        self.main_plot.addItem(self.img_item)
        
        self.square_line = pg.PlotDataItem(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.DashLine))
        self.main_plot.addItem(self.square_line)
        
        self.center_point = pg.ScatterPlotItem(pen=pg.mkPen('r'), brush=pg.mkBrush('r'), size=5)
        self.main_plot.addItem(self.center_point)
        
        self.setpoint_marker = pg.ScatterPlotItem(pen=pg.mkPen('g'), brush=pg.mkBrush('g'), size=7)
        self.main_plot.addItem(self.setpoint_marker)
        
        self.points_scatter = pg.ScatterPlotItem(pen=pg.mkPen('b'), brush=pg.mkBrush('b'), size=3)
        self.main_plot.addItem(self.points_scatter)
        
        self.outer_circle = None
        self.inner_circle = None
        
        # Enable mouse interaction for panning and zooming
        self.main_plot.setMouseEnabled(x=True, y=True)
        self.residuals_plot.setMouseEnabled(x=True, y=True)

        # Add variables to track manual view state for both plots
        self.main_plot_zoom_active = False
        self.residuals_plot_zoom_active = False
        self.initial_main_view_set = False
        self.initial_residuals_view_set = False
        
        # Connect signals to detect user interaction with the views
        self.main_plot.sigRangeChanged.connect(self.main_view_changed)
        self.residuals_plot.sigRangeChanged.connect(self.residuals_view_changed)

        # Set up a timer for periodic updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)  # Update every 100 ms

        # Flag to track if the window has been closed
        self.closed = False
        
        # Show the widget
        self.show()
        
        # Set the aspect ratio to be equal (square axes)
        self.main_plot.setAspectLocked(True)
        self.residuals_plot.setAspectLocked(True)
        
        # Create a jet colormap
        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.jet_colormap = ColorMap(pos=positions, color=colors)
        self.img_item.setLookupTable(self.jet_colormap.getLookupTable(0.0, 1.0, 256))
        
    def closeEvent(self, event):
        """Handle window close event - ensures cleanup"""
        self.closed = True
        self.timer.stop()
        event.accept()
        
    def close(self):
        """Explicitly close the widget and stop the timer"""
        self.closed = True
        self.timer.stop()
        super().close()

    # Function to track when user changes the main view
    def main_view_changed(self, view):
        self.main_plot_zoom_active = True
    
    # Function to track when user changes the residuals view
    def residuals_view_changed(self, view):
        self.residuals_plot_zoom_active = True
    
    def update_plot(self):
        """Update the plot periodically"""
        if not self.closed:
            # Process events to update the plot and allow interaction
            QtWidgets.QApplication.processEvents()

    def update(self, image, center, side, theta,
               setpoint=None, points=None, radius=None, radtol=None,
               centerguess=None, zoom_factor=2.5, 
               cmap='jet', title=None):
        """Update the plot with new data"""
        if self.closed:
            return
            
        c_y, c_x = center
        half = side / 2.0
        
        # Track center history
        self.centers_history.append(center)
        
        # Handle setpoint updates
        if setpoint is not None:
            self.setpoints_history.append(setpoint)
        elif len(self.setpoints_history) > 0:
            self.setpoints_history.append(self.setpoints_history[-1])
        else:
            self.setpoints_history.append(center)
        
        current_setpoint = self.setpoints_history[-1]
        
        # Define corners in canonical (unrotated) coordinates
        corners = np.array([
            [-half, -half],
            [-half, half],
            [half, half],
            [half, -half],
            [-half, -half]  # to close the square
        ])
        
        # Apply rotation
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated = corners @ rot.T
        square_yx = rotated + [c_y, c_x]
        
        # Update image
        self.img_item.setImage(image.T, autoLevels=True)
        
        # Update square
        self.square_line.setData(square_yx[:, 1], square_yx[:, 0])
        
        # Update fitted center
        self.center_point.setData([c_x], [c_y])
        
        # Update setpoint marker
        self.setpoint_marker.setData([current_setpoint[1]], [current_setpoint[0]])
        
        # Update observed points
        if points is not None and len(points) > 0:
            points = np.array(points)
            self.points_scatter.setData(points[:, 1], points[:, 0])
        
        # Update annulus
        if radius is not None and radtol is not None:
            annulus_center = centerguess if centerguess is not None else (c_y, c_x)
            sc_y, sc_x = annulus_center
            if self.outer_circle is None:
                self.outer_circle = pg.CircleROI([sc_x - radius - radtol, sc_y - radius - radtol], 
                                                 [2 * (radius + radtol), 2 * (radius + radtol)], 
                                                 pen=pg.mkPen('c', style=pg.QtCore.Qt.DashLine))
                self.main_plot.addItem(self.outer_circle)
            else:
                self.outer_circle.setPos([sc_x - radius - radtol, sc_y - radius - radtol])
                self.outer_circle.setSize([2 * (radius + radtol), 2 * (radius + radtol)])
            
            if self.inner_circle is None:
                self.inner_circle = pg.CircleROI([sc_x - radius + radtol, sc_y - radius + radtol], 
                                                 [2 * (radius - radtol), 2 * (radius - radtol)], 
                                                 pen=pg.mkPen('c', style=pg.QtCore.Qt.DashLine))
                self.main_plot.addItem(self.inner_circle)
            else:
                self.inner_circle.setPos([sc_x - radius + radtol, sc_y - radius + radtol])
                self.inner_circle.setSize([2 * (radius - radtol), 2 * (radius - radtol)])
        
        # Only set the main plot view range if this is the first update or user hasn't zoomed/panned
        if not self.initial_main_view_set or not self.main_plot_zoom_active:
            zoom_half = (side * zoom_factor) / 2
            self.main_plot.setXRange(c_x - zoom_half, c_x + zoom_half)
            self.main_plot.setYRange(c_y - zoom_half, c_y + zoom_half)
            self.initial_main_view_set = True
        
        # Update residuals plot with actual values
        if len(self.centers_history) > 0 and len(self.setpoints_history) > 0:
            # Plot the setpoint as an 'X'
            self.residuals_plot.clear()
            self.residuals_plot.plot([current_setpoint[1]], [current_setpoint[0]], pen=None, symbol='x', symbolBrush='g', symbolSize=10)
            
            # Plot the actual measured centers
            x_centers = [c[1] for c in self.centers_history]
            y_centers = [c[0] for c in self.centers_history]
            self.residuals_plot.plot(x_centers, y_centers, pen=pg.mkPen('b', width=1), symbol='o', symbolBrush='b', symbolSize=5)

            # Set equal aspect ratio
            self.residuals_plot.setAspectLocked(True)

            # Only set the residuals plot view range if user hasn't manually zoomed/panned
            if not self.initial_residuals_view_set or not self.residuals_plot_zoom_active:
                min_x = min(x_centers + [current_setpoint[1]])
                max_x = max(x_centers + [current_setpoint[1]])
                min_y = min(y_centers + [current_setpoint[0]])
                max_y = max(y_centers + [current_setpoint[0]])
                
                # Add some padding to the ranges
                x_padding = max(10, (max_x - min_x) * 0.1)
                y_padding = max(10, (max_y - min_y) * 0.1)
                
                self.residuals_plot.setXRange(min_x - x_padding, max_x + x_padding)
                self.residuals_plot.setYRange(min_y - y_padding, max_y + y_padding)
                self.initial_residuals_view_set = True

            # Calculate and display the current residual error
            current_residual_error = np.sqrt((x_centers[-1] - current_setpoint[1])**2 + (y_centers[-1] - current_setpoint[0])**2)
            self.residual_label.setText(f"Residual Error: {current_residual_error:.2f} px")

        # Process events to update the plot and allow interaction
        if not self.closed:
            QtWidgets.QApplication.processEvents()
        
    # Method to reset zoom to default for all plots
    def reset_zoom(self):
        """Reset zoom to default for all plots - can be connected to a reset button if needed."""
        self.main_plot_zoom_active = False
        self.residuals_plot_zoom_active = False

    def execute(self):
        """Execute the application event loop"""
        if not self.closed and hasattr(self, 'app'):
            self.app.exec_()
