import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, QFileDialog)
from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QKeySequence
import pyqtgraph as pg


class NumpyArrayViewer(QMainWindow):
    def __init__(self, data=None, user_points=None):
        super().__init__()
        
        # Initialize with provided data or create empty array
        if data is not None:
            self.data = data
        else:
            # Default sample data if none provided
            self.data = np.random.rand(500, 500)
        
        # Store selected points
        self.selected_points = []
        
        # Store user-supplied points
        self.user_points = user_points if user_points is not None else []
        
        # Create QApplication instance if it doesn't exist
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
            self.owns_app = True
        else:
            self.owns_app = False
            
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Shift click the generated speckles, then close')
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Create pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)
        
        # Add histogram for controlling color mapping
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_item)
        self.plot_widget.addItem(self.hist)
        
        # Set viewbox for zooming and panning
        self.viewbox = self.plot_widget.getPlotItem().getViewBox()
        self.viewbox.setMouseMode(pg.ViewBox.RectMode)
        
        # Connect signals for zoom and pan
        self.viewbox.sigRangeChanged.connect(self.range_changed)
        
        # Connect mouse click for point selection (when Shift is pressed)
        self.plot_widget.scene().sigMouseClicked.connect(self.mouse_clicked)
        
        # Scatter plot for marking selected points (red X)
        self.scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen('r', width=2), symbol='x')
        self.plot_widget.addItem(self.scatter)
        
        # Scatter plot for user-supplied points (green X)
        self.user_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen('g', width=2), symbol='x')
        self.plot_widget.addItem(self.user_scatter)
        
        # Add crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        
        # Mouse movement proxy for updating crosshair and value display
        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)
        
        # Control panel
        controls_layout = QHBoxLayout()
        
        # Reset view button
        reset_button = QPushButton("Reset View")
        reset_button.clicked.connect(self.reset_view)
        controls_layout.addWidget(reset_button)
        
        # Clear points button
        clear_points_button = QPushButton("Clear Selected Points")
        clear_points_button.clicked.connect(self.clear_selected_points)
        controls_layout.addWidget(clear_points_button)
        
        # Toggle user points visibility
        self.toggle_user_points_button = QPushButton("Hide User Points")
        self.toggle_user_points_button.clicked.connect(self.toggle_user_points)
        controls_layout.addWidget(self.toggle_user_points_button)
        self.user_points_visible = True
        
        # Value at cursor
        self.value_label = QLabel("Value: --")
        controls_layout.addWidget(self.value_label)
        
        # Position display
        self.pos_label = QLabel("Position: (--,--)")
        controls_layout.addWidget(self.pos_label)
        
        # Add controls to main layout
        main_layout.addWidget(self.plot_widget)
        main_layout.addLayout(controls_layout)
        
        # Setup timer for processing events in non-blocking mode
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_events)
        self.timer.start(50)  # Process events every 50ms
        
        # Display the initial data
        self.update_display()
        
    def update_display(self):
        # Update the image with current data
        self.img_item.setImage(self.data.T)  # Transpose the data for correct display
        self.viewbox.autoRange()  # Automatically adjust range to show all data
        
        # Update the scatter plots
        self.update_scatter_plot()
        self.update_user_scatter_plot()
    
    def update_data(self, new_data):
        """Update the displayed data without blocking execution"""
        self.data = new_data
        self.img_item.setImage(self.data.T)  # Transpose the data for correct display
        # Process events to update the display
        self.app.processEvents()
        
    def process_events(self):
        """Process Qt events to keep the UI responsive"""
        self.app.processEvents()
        
    def mouse_clicked(self, event):
        # Check if shift key is pressed
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            # Get the position in data coordinates
            pos = event.scenePos()
            mouse_point = self.viewbox.mapSceneToView(pos)
            x, y = int(mouse_point.x()), int(mouse_point.y())
            
            # Check if within data bounds (note: data is transposed in display)
            if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                # Get value from the original (non-transposed) data
                value = self.data[y, x]
                
                # Store the point as (x, y) - matplotlib convention
                self.selected_points.append((x, y, value))
                print(f"Selected point: ({x}, {y}) with value {value:.4f}")
                
                # Update the scatter plot to show markers
                self.update_scatter_plot()
    
    def update_scatter_plot(self):
        # Update the scatter plot with all selected points
        positions = [(x, y) for x, y, _ in self.selected_points]
        if positions:
            self.scatter.setData(pos=positions)
    
    def update_user_scatter_plot(self):
        # Update the scatter plot with user-supplied points
        if self.user_points and self.user_points_visible:
            # Convert to list of (x,y) tuples if it's a numpy array
            if isinstance(self.user_points, np.ndarray):
                positions = [(x, y) for x, y in self.user_points]
            else:
                positions = self.user_points
            self.user_scatter.setData(pos=positions)
        else:
            self.user_scatter.setData(pos=[])
    
    def toggle_user_points(self):
        # Toggle visibility of user-supplied points
        self.user_points_visible = not self.user_points_visible
        if self.user_points_visible:
            self.toggle_user_points_button.setText("Hide User Points")
            self.update_user_scatter_plot()
        else:
            self.toggle_user_points_button.setText("Show User Points")
            self.user_scatter.setData(pos=[])
    
    def set_user_points(self, points):
        # Method to update user points from outside
        self.user_points = points
        self.update_user_scatter_plot()
        # Process events to update the display
        self.app.processEvents()
    
    def clear_selected_points(self):
        # Clear all selected points
        self.selected_points = []
        self.scatter.setData(pos=[])
        print("Cleared all selected points")
        
    def mouse_moved(self, evt):
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            # Get mouse position in data coordinates
            mouse_point = self.viewbox.mapSceneToView(pos)
            x, y = int(mouse_point.x()), int(mouse_point.y())
            
            # Update crosshair
            self.vLine.setPos(x)
            self.hLine.setPos(y)
            
            # Update label with value and position
            if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                value = self.data[y, x]  # Get value from original data
                self.value_label.setText(f"Value: {value:.4f}")
                self.pos_label.setText(f"Position: ({x},{y})")
            else:
                self.value_label.setText("Value: --")
                self.pos_label.setText("Position: (--,--)")
    
    def range_changed(self, viewbox, ranges):
        # This gets called when zoom or pan changes
        pass  # You can add custom behavior here if needed
    
    def reset_view(self):
        # Reset to default view
        self.viewbox.autoRange()
    
    def load_data(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open NumPy Array", "", "NumPy Files (*.npy);;All Files (*)")
        if filename:
            try:
                new_data = np.load(filename)
                self.data = new_data
                self.update_display()
                # Clear selected points when loading new data
                self.clear_selected_points()
            except Exception as e:
                print(f"Error loading file: {e}")
                
    def closeEvent(self, event):
        # This method is called when the window is closed
        if self.selected_points:
            print("\nSelected Points Summary:")
            print("------------------------")
            for i, (x, y, value) in enumerate(self.selected_points):
                print(f"Point {i+1}: Coordinates ({x}, {y})")
        
        # Stop the timer
        self.timer.stop()
        
        # Call the parent class closeEvent
        super().closeEvent(event)


def generate_sample_data(size=500):
    """Generate a sample 2D NumPy array with some interesting patterns."""
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Create some interesting patterns
    z = np.sin(0.5 * x_grid) * np.cos(0.5 * y_grid)
    z += 0.2 * np.sin(x_grid * 2) * np.cos(y_grid * 2)
    z += np.exp(-(x_grid**2 + y_grid**2) / 20)
    
    return z

def run_viewer(data=None, user_points=None):
    """
    Run the NumPy array viewer and return the selected points when closed.
    
    Args:
        data (numpy.ndarray, optional): NumPy array to display initially
        user_points (list or numpy.ndarray, optional): List of (x,y) tuples or Nx2 array to display as green crosses
        
    Returns:
        list: List of tuples containing (x, y) coordinates of selected points
    """
    # Start the application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True
    else:
        owns_app = False
    
    # Create viewer with data and user points
    if data is not None:
        viewer = NumpyArrayViewer(data, user_points)
    else:
        sample_data = generate_sample_data()
        viewer = NumpyArrayViewer(sample_data, user_points)
        
    viewer.show()
    
    # Run the application
    if owns_app:
        app.exec_()
    
    # Return only the coordinates (x, y) of the selected points
    return [(x, y) for x, y, _ in viewer.selected_points]

def create_non_blocking_viewer(data=None, user_points=None):
    """
    Create a non-blocking NumPy array viewer that can be updated from outside.
    
    Args:
        data (numpy.ndarray, optional): NumPy array to display initially
        user_points (list or numpy.ndarray, optional): List of (x,y) tuples or Nx2 array to display as green crosses
        
    Returns:
        NumpyArrayViewer: The viewer instance that can be updated
    """
    # Create viewer with data and user points
    if data is not None:
        viewer = NumpyArrayViewer(data, user_points)
    else:
        sample_data = generate_sample_data()
        viewer = NumpyArrayViewer(sample_data, user_points)
        
    viewer.show()
    
    # Process events to make the window appear
    QApplication.instance().processEvents()
    
    return viewer

if __name__ == '__main__':
    # Create a sample data array and save it
    sample_data = generate_sample_data()
    np.save('sample_array.npy', sample_data)
    print("Saved sample data to 'sample_array.npy'")
    
    # Create some sample user points
    sample_user_points = [(100, 100), (200, 200), (300, 300), (400, 400)]
    
    # Run the viewer and get the selected coordinates
    selected_coordinates = run_viewer(data=sample_data, user_points=sample_user_points)
    
    # Use the selected coordinates
    print("\nReturned selected coordinates:", selected_coordinates)
