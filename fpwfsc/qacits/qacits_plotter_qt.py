import sys
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QTransform
import pyqtgraph as pg
from pyqtgraph import ColorMap

class QacitsPlotter(QtWidgets.QWidget):
    def __init__(self, figsize=(600, 600)):
        # Initialize the QApplication if it hasn't been created yet
        if not QtWidgets.QApplication.instance():
            self.app = QtWidgets.QApplication(sys.argv)
        else:
            self.app = QtWidgets.QApplication.instance()

        super().__init__()

        # Set up the window
        self.setWindowTitle("Live Image Plotter")
        self.resize(int(figsize[0]), int(figsize[1]))

        # Create a main layout
        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)

        # Create the main image plot
        self.main_plot = pg.PlotWidget()
        main_layout.addWidget(self.main_plot)

        # Create stretch control panel (vertical stacking)
        stretch_panel = QtWidgets.QWidget()
        stretch_layout = QtWidgets.QVBoxLayout(stretch_panel)
        stretch_layout.setContentsMargins(5, 2, 5, 2)
        stretch_layout.setSpacing(2)
        
        # Lower percentile slider row
        lower_row = QtWidgets.QWidget()
        lower_layout = QtWidgets.QHBoxLayout(lower_row)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        
        lower_layout.addWidget(QtWidgets.QLabel("Min:"))
        self.lower_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.lower_slider.setMinimum(0)
        self.lower_slider.setMaximum(50)
        self.lower_slider.setValue(1)  # Default 1st percentile
        self.lower_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.lower_slider.setTickInterval(10)
        self.lower_slider.valueChanged.connect(self.on_stretch_changed)
        lower_layout.addWidget(self.lower_slider)
        
        self.lower_label = QtWidgets.QLabel("1%")
        self.lower_label.setMinimumWidth(40)
        lower_layout.addWidget(self.lower_label)
        
        stretch_layout.addWidget(lower_row)
        
        # Upper percentile slider row
        upper_row = QtWidgets.QWidget()
        upper_layout = QtWidgets.QHBoxLayout(upper_row)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        
        upper_layout.addWidget(QtWidgets.QLabel("Max:"))
        self.upper_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.upper_slider.setMinimum(50)
        self.upper_slider.setMaximum(100)
        self.upper_slider.setValue(99)  # Default 99th percentile
        self.upper_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.upper_slider.setTickInterval(10)
        self.upper_slider.valueChanged.connect(self.on_stretch_changed)
        upper_layout.addWidget(self.upper_slider)
        
        self.upper_label = QtWidgets.QLabel("99%")
        self.upper_label.setMinimumWidth(40)
        upper_layout.addWidget(self.upper_label)
        
        stretch_layout.addWidget(upper_row)
        
        main_layout.addWidget(stretch_panel)

        # Initialize plot elements
        self.img_item = pg.ImageItem()
        self.main_plot.addItem(self.img_item)

        # Store current image for percentile calculations
        self.current_image = None

        # Initialize overlay elements
        self.inner_circle = pg.CircleROI([0, 0], [0, 0], pen=pg.mkPen('r', width=4), movable=False)
        self.outer_circle = pg.CircleROI([0, 0], [0, 0], pen=pg.mkPen('r', width=4), movable=False)

        # Create line segments instead of infinite lines
        self.h_line_top = pg.PlotDataItem(pen=pg.mkPen('r', width=4, style=Qt.DashLine))
        self.h_line_bottom = pg.PlotDataItem(pen=pg.mkPen('r', width=4, style=Qt.DashLine))
        self.v_line_left = pg.PlotDataItem(pen=pg.mkPen('r', width=4, style=Qt.DashLine))
        self.v_line_right = pg.PlotDataItem(pen=pg.mkPen('r', width=4, style=Qt.DashLine))

        self.center_point = pg.ScatterPlotItem([0], [0], symbol='+', size=15, pen=pg.mkPen('r', width=4))

        # Add overlay elements to plot (initially hidden)
        self.main_plot.addItem(self.inner_circle)
        self.main_plot.addItem(self.outer_circle)
        self.main_plot.addItem(self.h_line_top)
        self.main_plot.addItem(self.h_line_bottom)
        self.main_plot.addItem(self.v_line_left)
        self.main_plot.addItem(self.v_line_right)
        self.main_plot.addItem(self.center_point)

        # Hide overlays initially
        self.inner_circle.hide()
        self.outer_circle.hide()
        self.h_line_top.hide()
        self.h_line_bottom.hide()
        self.v_line_left.hide()
        self.v_line_right.hide()
        self.center_point.hide()

        # Enable mouse interaction for panning and zooming
        self.main_plot.setMouseEnabled(x=True, y=True)

        # Pixel value readout on hover
        self.pixel_label = QtWidgets.QLabel("")
        self.pixel_label.setStyleSheet("color: white; background: rgba(0,0,0,150); padding: 2px;")
        main_layout.addWidget(self.pixel_label)
        self._img_x_min = 0
        self._img_y_min = 0
        self._img_dx = 1
        self._img_dy = 1
        self.proxy = pg.SignalProxy(self.main_plot.scene().sigMouseMoved,
                                   rateLimit=30, slot=self._on_mouse_moved)

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

        # Create a jet colormap
        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.jet_colormap = ColorMap(pos=positions, color=colors)
        self.img_item.setLookupTable(self.jet_colormap.getLookupTable(0.0, 1.0, 256))

    def on_stretch_changed(self):
        """Handle slider value changes"""
        # Update labels
        lower_val = self.lower_slider.value()
        upper_val = self.upper_slider.value()
        self.lower_label.setText(f"{lower_val}%")
        self.upper_label.setText(f"{upper_val}%")
        
        # Update image levels
        self.update_image_levels()

    def update_image_levels(self):
        """Update the image display levels based on current slider values"""
        if self.current_image is None:
            return
        
        lower_percentile = self.lower_slider.value()
        upper_percentile = self.upper_slider.value()
        
        # Ensure lower < upper
        if lower_percentile >= upper_percentile:
            return
        
        # Calculate intensity levels
        vmin = np.percentile(self.current_image, lower_percentile)
        vmax = np.percentile(self.current_image, upper_percentile)
        
        # Apply levels to existing image
        self.img_item.setLevels([vmin, vmax])

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

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if self.current_image is None:
            return
        mouse_point = self.main_plot.plotItem.vb.mapSceneToView(pos)
        mx, my = mouse_point.x(), mouse_point.y()
        col = int(round((mx - self._img_x_min) / self._img_dx))
        row = int(round((my - self._img_y_min) / self._img_dy))
        h, w = self.current_image.shape[:2]
        if 0 <= row < h and 0 <= col < w:
            val = self.current_image[row, col]
            self.pixel_label.setText(f"x={mx:.1f}  y={my:.1f}  val={val:.2f}")
        else:
            self.pixel_label.setText("")

    def update_plot(self):
        """Update the plot periodically"""
        if not self.closed:
            # Process events to update the plot and allow interaction
            QtWidgets.QApplication.processEvents()

    def update(self, image=None, x_center=None, y_center=None, min_radius=None, max_radius=None,
               x_coords=None, y_coords=None, title=None):
        """
        Update the plot with new image data and quad cell overlay

        Parameters:
        -----------
        image : numpy.ndarray
            Input image to display
        x_center : float, optional
            X coordinate of center point
        y_center : float, optional
            Y coordinate of center point
        min_radius : float, optional
            Inner radius of annulus
        max_radius : float, optional
            Outer radius of annulus
        x_coords : numpy.ndarray, optional
            2D array of x coordinates (same shape as image). If None, uses pixel indices.
        y_coords : numpy.ndarray, optional
            2D array of y coordinates (same shape as image). If None, uses pixel indices.
        title : str, optional
            Title for the plot
        """
        if self.closed or image is None:
            return

        # Store current image for stretch calculations
        self.current_image = image.copy()

        # Calculate levels before setting image
        lower_percentile = self.lower_slider.value()
        upper_percentile = self.upper_slider.value()
        
        if lower_percentile < upper_percentile:
            vmin = np.percentile(self.current_image, lower_percentile)
            vmax = np.percentile(self.current_image, upper_percentile)
        else:
            # Fallback if sliders are in invalid state
            vmin = np.min(self.current_image)
            vmax = np.max(self.current_image)

        # Create coordinate arrays if not provided
        if x_coords is None or y_coords is None:
            height, width = image.shape[:2]
            y_indices, x_indices = np.mgrid[0:height, 0:width]
            x_coords = x_indices if x_coords is None else x_coords
            y_coords = y_indices if y_coords is None else y_coords

        # Set up image extent for proper coordinate display
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        self._img_x_min = x_min
        self._img_y_min = y_min
        self._img_dx = (x_max - x_min) / image.shape[1] if image.shape[1] > 1 else 1
        self._img_dy = (y_max - y_min) / image.shape[0] if image.shape[0] > 1 else 1

        if title is not None:
            self.main_plot.setTitle(title)

        # Update image with calculated levels (transpose for correct orientation in pyqtgraph)
        self.img_item.setImage(image.T, autoLevels=False, levels=[vmin, vmax])

        # Set the position and scale of the image to match coordinates
        self.img_item.setPos(x_min, y_min)
        dx = (x_max - x_min) / image.shape[1]
        dy = (y_max - y_min) / image.shape[0]

        # Use transform to set different x and y scales
        transform = QTransform()
        transform.scale(dx, dy)
        self.img_item.setTransform(transform)

        # Update overlays if center and radii are provided
        if x_center is not None and y_center is not None and min_radius is not None and max_radius is not None:
            # Update inner circle (hide if radius is zero to avoid division by zero in CircleROI)
            if min_radius > 0:
                self.inner_circle.setPos([x_center - min_radius, y_center - min_radius])
                self.inner_circle.setSize([2 * min_radius, 2 * min_radius])
                self.inner_circle.show()
            else:
                self.inner_circle.hide()

            # Update outer circle
            self.outer_circle.setPos([x_center - max_radius, y_center - max_radius])
            self.outer_circle.setSize([2 * max_radius, 2 * max_radius])
            self.outer_circle.show()

            # Update crosshair lines
            # Horizontal lines (only between inner and outer radius)
            self.h_line_top.setData([x_center - max_radius, x_center - min_radius], [y_center, y_center])
            self.h_line_bottom.setData([x_center + min_radius, x_center + max_radius], [y_center, y_center])

            # Vertical lines (only between inner and outer radius)
            self.v_line_left.setData([x_center, x_center], [y_center - max_radius, y_center - min_radius])
            self.v_line_right.setData([x_center, x_center], [y_center + min_radius, y_center + max_radius])

            self.h_line_top.show()
            self.h_line_bottom.show()
            self.v_line_left.show()
            self.v_line_right.show()

            # Update center point
            self.center_point.setData([x_center], [y_center])
            self.center_point.show()
        else:
            # Hide overlays if no center/radii provided
            self.inner_circle.hide()
            self.outer_circle.hide()
            self.h_line_top.hide()
            self.h_line_bottom.hide()
            self.v_line_left.hide()
            self.v_line_right.hide()
            self.center_point.hide()

        # Process events to update the plot and allow interaction
        if not self.closed:
            QtWidgets.QApplication.processEvents()

    def execute(self):
        """Execute the application event loop"""
        if not self.closed and hasattr(self, 'app'):
            self.app.exec_()


# Example usage
if __name__ == "__main__":
    # Create test data
    test_image = np.random.rand(100, 100) * 0.1

    # Add a bright spot
    y_grid, x_grid = np.mgrid[0:100, 0:100]
    gaussian = np.exp(-((x_grid - 55)**2 + (y_grid - 45)**2) / (2 * 5**2))
    test_image += gaussian

    # Create plotter
    plotter = QacitsPlotter()

    # Update with image and overlay
    plotter.update(image=test_image, x_center=50, y_center=50, min_radius=10, max_radius=25,
                   title="Quad Cell Overlay Example")

    # Run the application
    plotter.execute()