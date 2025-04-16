import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui
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

    def update(self, image, center, side, theta,
               setpoint=None, points=None, radius=None, tol=None,
               search_center=None, zoom_factor=2, 
               cmap='gray', title=None):
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
        if radius is not None and tol is not None:
            annulus_center = search_center if search_center is not None else (c_y, c_x)
            sc_y, sc_x = annulus_center
            if self.outer_circle is None:
                self.outer_circle = pg.CircleROI([sc_x - radius - tol, sc_y - radius - tol], 
                                                 [2 * (radius + tol), 2 * (radius + tol)], 
                                                 pen=pg.mkPen('c', style=pg.QtCore.Qt.DashLine))
                self.main_plot.addItem(self.outer_circle)
            else:
                self.outer_circle.setPos([sc_x - radius - tol, sc_y - radius - tol])
                self.outer_circle.setSize([2 * (radius + tol), 2 * (radius + tol)])
            
            if self.inner_circle is None:
                self.inner_circle = pg.CircleROI([sc_x - radius + tol, sc_y - radius + tol], 
                                                 [2 * (radius - tol), 2 * (radius - tol)], 
                                                 pen=pg.mkPen('c', style=pg.QtCore.Qt.DashLine))
                self.main_plot.addItem(self.inner_circle)
            else:
                self.inner_circle.setPos([sc_x - radius + tol, sc_y - radius + tol])
                self.inner_circle.setSize([2 * (radius - tol), 2 * (radius - tol)])
        
        # Zoom around the square center
        zoom_half = (side * zoom_factor) / 2
        self.main_plot.setXRange(c_x - zoom_half, c_x + zoom_half)
        self.main_plot.setYRange(c_y - zoom_half, c_y + zoom_half)
        
        # Update residuals plot
        if len(self.centers_history) > 0 and len(self.setpoints_history) > 0:
            residuals = []

            for i in range(len(self.centers_history)):
                center = self.centers_history[i]
                setpoint = self.setpoints_history[i]
                res_y = center[0] - setpoint[0]
                res_x = center[1] - setpoint[1]
                residuals.append((res_x, res_y))

            x_res = [r[0] for r in residuals]
            y_res = [r[1] for r in residuals]

            # Plot residuals as a trail
            self.residuals_plot.clear()
            self.residuals_plot.plot(x_res, y_res, pen=pg.mkPen('b', width=1))
            self.residuals_plot.plot([x_res[-1]], [y_res[-1]], pen=None, symbol='o', symbolBrush='r', symbolSize=5)

            # Set equal aspect ratio
            self.residuals_plot.setAspectLocked(True)

            # Adaptive scaling based on recent residuals
            window_size = 5
            recent_residuals = residuals[-window_size:] if len(residuals) > window_size else residuals
            recent_x = [r[0] for r in recent_residuals]
            recent_y = [r[1] for r in recent_residuals]

            recent_dists = [np.sqrt(x**2 + y**2) for x, y in zip(recent_x, recent_y)]
            recent_max_dist = max(recent_dists) if recent_dists else 0

            min_scale = 1.0
            axis_limit = max(min_scale, recent_max_dist * 1.5)

            self.residuals_plot.setXRange(-axis_limit, axis_limit)
            self.residuals_plot.setYRange(-axis_limit, axis_limit)

            # Calculate and display the current residual error
            current_residual_error = np.sqrt(x_res[-1]**2 + y_res[-1]**2)
            self.residual_label.setText(f"Residual Error: {current_residual_error:.2f} px")

        # Process events to update the plot
        QtWidgets.QApplication.processEvents()

    def execute(self):
        # Execute the application
        if hasattr(self, 'app'):
            self.app.exec_()
