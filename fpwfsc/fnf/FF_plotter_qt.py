import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, pyqtSlot, QObject
import pyqtgraph as pg


class PlotterSignals(QObject):
    """
    Signal class for thread-safe communication with the plotter.
    This allows data to be safely passed from worker threads to the GUI.
    """
    update_signal = pyqtSignal(object, object, object, object, object, object)


class LivePlotter(QtWidgets.QWidget):
    """
    PyQt5-based plotter for Fast and Furious optical control visualization.
    Replaces the matplotlib LivePlotter in plotting_funcs.py with a more
    responsive and less buggy implementation using PyQtGraph.
    """
    def __init__(self, figsize=(800, 800)):
        # Initialize the QApplication if it hasn't been created yet
        if not QtWidgets.QApplication.instance():
            self.app = QtWidgets.QApplication(sys.argv)
        else:
            self.app = QtWidgets.QApplication.instance()
        
        super().__init__()
        
        # Create signals for thread-safe communication
        self.signals = PlotterSignals()
        self.signals.update_signal.connect(self._update_plots)
        
        # Set up the window
        self.setWindowTitle("Fast and Furious - Live Plotter")
        self.resize(figsize[0], figsize[1])
        
        # Create a layout
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        
        # Create the PSF plot (top left)
        self.psf_plot = pg.PlotWidget()
        self.psf_plot.setTitle("PSF")
        layout.addWidget(self.psf_plot, 0, 0)
        
        # Create the wavefront residuals plot (top right)
        self.wf_plot = pg.PlotWidget()
        self.wf_plot.setTitle("Wavefront Residuals")
        layout.addWidget(self.wf_plot, 0, 1)
        
        # Create the Strehl ratio plot (bottom left)
        self.strehl_plot = pg.PlotWidget()
        self.strehl_plot.setTitle("Strehl Ratio")
        self.strehl_plot.setLabel('left', 'Strehl')
        self.strehl_plot.setLabel('bottom', 'Iteration')
        self.strehl_plot.showGrid(x=True, y=True)
        layout.addWidget(self.strehl_plot, 1, 0)
        
        # Create the VAR plot (bottom right)
        self.var_plot = pg.PlotWidget()
        self.var_plot.setTitle("VAR")
        self.var_plot.setLabel('left', 'VAR')
        self.var_plot.setLabel('bottom', 'Iteration')
        self.var_plot.showGrid(x=True, y=True)
        self.var_plot.setLogMode(y=True)  # Log scale for y-axis
        # Explicitly set the y-range
        self.var_plot.getViewBox().setYRange(np.log10(1e-4), np.log10(1), padding=0)
        layout.addWidget(self.var_plot, 1, 1)
        
        # Initialize plot items
        self.psf_img = pg.ImageItem()
        self.psf_plot.addItem(self.psf_img)
        
        self.wf_img = pg.ImageItem()
        self.wf_plot.addItem(self.wf_img)
        
        self.strehl_curve = pg.PlotDataItem(pen=pg.mkPen('g', width=2))
        self.strehl_plot.addItem(self.strehl_curve)
        
        self.var_curve = pg.PlotDataItem(pen=pg.mkPen('r', width=2))
        self.var_plot.addItem(self.var_curve)
        
        # Create colormaps
        # Jet colormap for PSF
        jet_colors = [
            (0, 0, 127),
            (0, 0, 255),
            (0, 255, 255),
            (255, 255, 0),
            (255, 0, 0)
        ]
        pos = np.linspace(0, 1, len(jet_colors))
        
        # Create lookup tables directly - more compatible with different PyQtGraph versions
        self.psf_lut = self._create_colormap(jet_colors, pos)
        self.psf_img.setLookupTable(self.psf_lut)
        
        # Blue-White-Red colormap for wavefront residuals
        bwr_colors = [
            (0, 0, 255),
            (255, 255, 255),
            (255, 0, 0)
        ]
        pos_bwr = np.linspace(0, 1, len(bwr_colors))
        self.wf_lut = self._create_colormap(bwr_colors, pos_bwr)
        self.wf_img.setLookupTable(self.wf_lut)
        
        # Set axis behavior
        self.psf_plot.setAspectLocked(True)
        self.wf_plot.setAspectLocked(True)
        
        # Set up y-axis limits
        self.strehl_plot.setYRange(0, 1.1)
        self.var_plot.setYRange(1e-4, 1)
        
        # Set up a timer for periodic updates (process Qt events)
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_events)
        self.timer.start(50)  # Update every 50 ms
        
        # Flag to track if the window has been closed
        self.closed = False
        
        # Show the widget
        self.show()
    
    def _create_colormap(self, colors, positions):
        """
        Create a colormap lookup table directly.
        More compatible with different PyQtGraph versions.
        
        Parameters:
        -----------
        colors : list of tuples
            List of RGB color tuples with values from 0-255
        positions : ndarray
            Normalized positions (0-1) for each color
            
        Returns:
        --------
        lut : ndarray
            Lookup table for the colormap
        """
        nsteps = 256
        lut = np.zeros((nsteps, 3), dtype=np.uint8)
        
        # Interpolate the colors
        for i in range(3):  # R, G, B
            channel_values = [colors[j][i] for j in range(len(colors))]
            lut[:, i] = np.interp(np.linspace(0, 1, nsteps), positions, channel_values)
            
        return lut
    
    def _ensure_2d(self, data):
        """
        Ensure data is 2D, converting from 1D if necessary.
        This mimics HCIPy's imshow_field behavior of reshaping 1D arrays to 2D.
        
        Parameters:
        -----------
        data : ndarray
            Input data array (1D or 2D)
            
        Returns:
        --------
        data_2d : ndarray
            2D array (reshaped if input was 1D)
        """
        if data is None:
            return np.zeros((1, 1))
            
        # Convert to numpy array if not already
        data_array = np.asarray(data)
        
        # Handle shaped field objects from HCIPy (if they have a .shaped attribute)
        if hasattr(data_array, 'shaped'):
            return np.asarray(data_array.shaped)
            
        # Handle complex arrays (from FFT operations)
        if np.iscomplexobj(data_array):
            data_array = np.abs(data_array)
        
        # If data is 1D, reshape to square 2D
        if data_array.ndim == 1:
            n = int(np.sqrt(len(data_array)))
            if n*n != len(data_array):
                # If not a perfect square, pad with zeros
                next_square = (n+1)**2
                padded = np.zeros(next_square)
                padded[:len(data_array)] = data_array
                return padded.reshape((n+1, n+1))
            
            return data_array.reshape((n, n)) #to make consistent with matplotlib
            
        # If array has more than 2 dimensions, take first 2D slice
        if data_array.ndim > 2:
            return data_array[..., 0]
            
        return data_array
    
    def closeEvent(self, event):
        """Handle window close event - ensures proper cleanup"""
        self.closed = True
        self.timer.stop()
        event.accept()
    
    def close(self):
        """Explicitly close the widget and stop the timer"""
        self.closed = True
        self.timer.stop()
        super().close()
    
    def process_events(self):
        """Process events to keep the UI responsive"""
        if not self.closed:
            QtWidgets.QApplication.processEvents()
    
    def update(self, Niter, data, pupil_wf, aperture, SRA, VAR):
        """
        Thread-safe update method. Emits a signal that's connected to _update_plots.
        This method can be safely called from any thread.
        
        Parameters:
        -----------
        Niter : int
            Total number of iterations
        data : ndarray
            The PSF image data
        pupil_wf : ndarray
            The wavefront residuals at the pupil
        aperture : ndarray
            The aperture mask
        SRA : ndarray
            Strehl ratio array (history)
        VAR : ndarray
            Variance array (history)
        """
        if self.closed:
            return
            
        # Make copies of the numpy arrays to avoid threading issues
        if data is not None:
            data_copy = np.array(data)
        else:
            data_copy = None
            
        if pupil_wf is not None:
            pupil_wf_copy = np.array(pupil_wf)
        else:
            pupil_wf_copy = None
            
        if aperture is not None:
            aperture_copy = np.array(aperture)
        else:
            aperture_copy = None
            
        SRA_copy = np.array(SRA)
        VAR_copy = np.array(VAR)
        
        # Emit the signal with the data to update the plots in the main thread
        self.signals.update_signal.emit(Niter, data_copy, pupil_wf_copy, aperture_copy, SRA_copy, VAR_copy)
    
    @pyqtSlot(object, object, object, object, object, object)
    def _update_plots(self, Niter, data, pupil_wf, aperture, SRA, VAR):
        """
        Actual update method that runs in the GUI thread.
        This is called via the signal connection and is thread-safe.
        
        Parameters are the same as for update().
        """
        if self.closed:
            return
        
        try:
            # Fixed x-range based on Niter
            self.strehl_plot.setXRange(0, Niter-1)
            self.var_plot.setXRange(0, Niter-1)
            
            # Get current iteration
            i = np.sum(~np.isnan(SRA))
            
            # Ensure all data is properly shaped 2D
            data_2d = self._ensure_2d(data)
            #transpose the data to match matplotlib and pyqtgraph display. 
            data_transpose = data_2d.T

            #consider flipping x and y for pupil and aperature for pyqt display?
            pupil_wf_2d = self._ensure_2d(pupil_wf)
            pupil_wf_transpose = pupil_wf_2d.T

            aperture_2d = self._ensure_2d(aperture)
            aperture_transpose = aperture_2d.T


            
            # Update PSF display (log scale, similar to original)
            data_max = np.max(np.abs(data_transpose))
            if data_max > 0:  # Avoid division by zero
                
                psf_data = np.log10(np.abs(data_transpose) / data_max + 1e-8)
                # Scale to 0-1 range for display
                psf_min = np.min(psf_data)
                psf_max = np.max(psf_data)
                if psf_max > psf_min:  # Avoid division by zero
                    psf_norm = (psf_data - psf_min) / (psf_max - psf_min)
                    self.psf_img.setImage(psf_norm)
                    
            self.psf_plot.setTitle(f"PSF - Iteration {i}/{Niter}")
            
            # Update wavefront residuals
            # Mask the wavefront with the aperture
            masked_wf = pupil_wf_transpose * (aperture_transpose > 0)
            max_res = np.max(np.abs(masked_wf))
            if max_res > 0:  # Avoid division by zero
                # Scale to -1 to 1 range
                norm_wf = masked_wf / max_res
                # Then scale to 0 to 1 for display (will be mapped through colormap)
                wf_display = (norm_wf + 1) / 2
                self.wf_img.setImage(wf_display)
            self.wf_plot.setTitle("Wavefront Residuals")
            
            # Update Strehl ratio plot
            valid_indices = ~np.isnan(SRA)
            iterations = np.arange(Niter)[valid_indices]
            valid_sra = SRA[valid_indices]
            
            if len(valid_sra) > 0:
                self.strehl_curve.setData(iterations, valid_sra)
                current_strehl = valid_sra[-1] if len(valid_sra) > 0 else 0
                self.strehl_plot.setTitle(f"Strehl Ratio = {current_strehl:.3f}")
            
            # Update VAR plot
            valid_var = VAR[valid_indices]
            if len(valid_var) > 0:
                self.var_curve.setData(iterations, valid_var)
                current_var = valid_var[-1] if len(valid_var) > 0 else 0
                self.var_plot.setTitle(f"VAR = {current_var:.3f}")
                
                # Force Y range reset on each update to ensure it's properly set
                self.var_plot.getViewBox().setYRange(np.log10(1e-4), np.log10(1), padding=0)
        
        except Exception as e:
            print(f"Error updating plots: {e}")
            import traceback
            traceback.print_exc()
    
    def execute(self):
        """Execute the application event loop for standalone use"""
        if not self.closed and hasattr(self, 'app'):
            self.app.exec_()


if __name__ == "__main__":
    # Simple test to make sure the plotter works on its own
    import time
    
    plotter = LivePlotter()
    
    # Create some random test data
    test_iterations = 100
    test_data = np.random.random((100, 100))
    test_pupil_wf = np.random.random((100, 100)) * 2 - 1  # Range [-1, 1]
    test_aperture = np.ones((100, 100))
    test_aperture[40:60, 40:60] = 0  # Create a central obstruction
    
    test_SRA = np.zeros(test_iterations)
    test_VAR = np.zeros(test_iterations)
    
    # Fill with random test values and update in a loop
    for i in range(10):
        test_SRA[i] = 0.5 + i * 0.05  # Increasing Strehl
        test_VAR[i] = 0.1 / (i + 1)   # Decreasing VAR
        
        # Update with the test data
        plotter.update(test_iterations, test_data, test_pupil_wf, test_aperture, test_SRA, test_VAR)
        
        # Simulate some work being done
        time.sleep(0.5)
    
    # Run the application
    plotter.execute()