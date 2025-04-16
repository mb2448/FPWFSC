import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib
import gc  # For garbage collection
matplotlib.use('Qt5Agg')  # Use Qt backend for interactive plotting

# Smaller font sizes
plt.rcParams.update({
    'font.size': 7,        # Base font size
    'axes.labelsize': 7,   # Axis labels
    'axes.titlesize': 8,   # Subplot titles
    'xtick.labelsize': 6,  # X-axis tick labels
    'ytick.labelsize': 6,  # Y-axis tick labels
    'legend.fontsize': 6,  # Legend text
    'figure.titlesize': 9  # Figure title
})
class LiveSquarePlotter:
    def __init__(self, initial_setpoint=None, figsize=(6, 6)):
        """
        Initialize the interactive square plotter.
        
        Parameters:
            initial_setpoint (tuple): The initial target (y, x) center coordinates
            figsize (tuple): Figure size in inches (width, height)
        """
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=figsize)
        
        # Create main image subplot (larger, top)
        self.ax_main = self.fig.add_subplot(211)
        
        # Create residuals subplot (smaller, bottom)
        self.ax_residuals = self.fig.add_subplot(212, aspect='equal')
        
        # Initialize tracking variables
        self.centers_history = []
        self.setpoints_history = []
        if initial_setpoint is not None:
            self.setpoints_history.append(initial_setpoint)
        self.iteration_count = 0
        
        # Initialize plot elements
        self.img = None
        self.square_line = None
        self.center_point = None
        self.setpoint_marker = None
        self.points_scatter = None
        self.inner_circle = None
        self.outer_circle = None
        # Adjust subplot positions to add space between them
        self.fig.subplots_adjust(hspace=0.0, top=0.95, bottom=0.05, left=0.1, right=0.95)
        
        # Make the main image plot take up more space (70% vs 30%)
        pos_main = self.ax_main.get_position()
        pos_res = self.ax_residuals.get_position()
        self.ax_main.set_position([pos_main.x0, pos_res.y0 + 0.6 * pos_res.height, 
                                  pos_main.width, pos_main.height * 1.3])
        self.ax_residuals.set_position([pos_res.x0, pos_res.y0, 
                                        pos_res.width, pos_res.height * 0.6])
        self.window_size = 10  # Number of recent points to consider for scaling
        
    def update(self, image, center, side, theta,
           setpoint=None, points=None, radius=None, tol=None,
           search_center=None, zoom_factor=2, 
           cmap='gray', title=None):
        """
        Update the plot with new square parameters.
    
        Parameters:
            image (2D ndarray): Image to display
            center (tuple): (y, x) center of the square
            side (float): Side length of the square
            theta (float): Rotation angle (radians)
            setpoint (tuple): Optional new (y, x) target center coordinates
            points (list): Optional observed points to overlay
            radius (float): Radius of annular search region
            tol (float): Tolerance of annulus (defines thickness)
            search_center (tuple): (y, x) center of annular search region
            zoom_factor (float): Size of zoom box relative to square side
            cmap (str): Colormap for image display
            title (str): Plot title
        """
        c_y, c_x = center
        half = side / 2.0
    
        # Track center history
        self.centers_history.append(center)
        self.iteration_count += 1
    
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
        if self.img is None:
            self.img = self.ax_main.imshow(image, cmap=cmap, origin='lower')
        else:
            self.img.set_data(image)

        # Optionally set vmin and vmax for consistent color scaling
        self.img.set_clim(vmin=np.min(image), vmax=np.max(image)) 
        
        # Update square
        if self.square_line is None:
            self.square_line, = self.ax_main.plot(square_yx[:, 1], square_yx[:, 0], 
                                                  'r-', linewidth=2, alpha=0.4)
        else:
            self.square_line.set_data(square_yx[:, 1], square_yx[:, 0])
    
        # Update fitted center
        if self.center_point is None:
            self.center_point, = self.ax_main.plot(c_x, c_y, marker='x', color='red', 
                                                   markersize=3, alpha=0.4, zorder=10)
        else:
            self.center_point.set_data(c_x, c_y)
    
        # Update setpoint marker
        if self.setpoint_marker is None:
            self.setpoint_marker, = self.ax_main.plot(current_setpoint[1], current_setpoint[0], 
                                                      marker='+', color='green', 
                                                      markersize=5, alpha=0.5, zorder=10)
        else:
            self.setpoint_marker.set_data(current_setpoint[1], current_setpoint[0])
    
        # Update observed points
        if points is not None and len(points) > 0:
            points = np.array(points)
            if self.points_scatter is None:
                self.points_scatter, = self.ax_main.plot(points[:, 1], points[:, 0], 
                                                         'bx', markersize=3)
            else:
                self.points_scatter.set_data(points[:, 1], points[:, 0])
    
        # Update annulus
        if radius is not None and tol is not None:
            annulus_center = search_center if search_center is not None else (c_y, c_x)
            sc_y, sc_x = annulus_center
            if self.outer_circle is None:
                self.outer_circle = Circle((sc_x, sc_y), radius + tol, edgecolor='cyan',
                                           facecolor='none', linestyle='--', 
                                           linewidth=1.5, alpha=0.3)
                self.ax_main.add_patch(self.outer_circle)
            else:
                self.outer_circle.center = (sc_x, sc_y)
                self.outer_circle.radius = radius + tol
            
            if self.inner_circle is None:
                self.inner_circle = Circle((sc_x, sc_y), radius - tol, edgecolor='cyan',
                                           facecolor='none', linestyle='--', 
                                           linewidth=1.5, alpha=0.3)
                self.ax_main.add_patch(self.inner_circle)
            else:
                self.inner_circle.center = (sc_x, sc_y)
                self.inner_circle.radius = radius - tol
    
        # Zoom around the square center
        zoom_half = (side * zoom_factor) / 2
        self.ax_main.set_xlim(c_x - zoom_half, c_x + zoom_half)
        self.ax_main.set_ylim(c_y - zoom_half, c_y + zoom_half)
    
        # Set title and labels for main plot
        if title:
            self.ax_main.set_title(title, fontsize=8)
        self.ax_main.set_xlabel("X (col)")
        self.ax_main.set_ylabel("Y (row)")
        self.ax_main.grid(True, alpha=0.3)

        # Update residuals plot as a radial plot with setpoint at center
        if len(self.centers_history) > 0 and len(self.setpoints_history) > 0:
            residuals = []

            for i in range(len(self.centers_history)):
                center = self.centers_history[i]
                setpoint = self.setpoints_history[i]
                res_y = center[0] - setpoint[0]
                res_x = center[1] - setpoint[1]
                residuals.append((res_x, res_y))

            # Draw crosshairs
            self.ax_residuals.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
            self.ax_residuals.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

            # Draw target at center
            self.ax_residuals.plot(0, 0, 'go', markersize=4)

            # Plot residuals as a trail
            x_res = [r[0] for r in residuals]
            y_res = [r[1] for r in residuals]

            # Plot all points with diminishing alpha
            for i in range(len(x_res)):
                alpha = 0.05
                if i < len(x_res) - 1:
                    self.ax_residuals.plot([x_res[i], x_res[i+1]], [y_res[i], y_res[i+1]], 
                                           'b-', alpha=alpha, linewidth=0.7)
                self.ax_residuals.plot(x_res[i], y_res[i], 'bo', markersize=2 + 2 * (i / max(1, len(x_res) - 1)), 
                                       alpha=alpha)

            # Highlight current position
            self.ax_residuals.plot(x_res[-1], y_res[-1], 'ro', markersize=1, alpha=0.8)

            # Set equal aspect ratio
            self.ax_residuals.set_aspect('equal')

            # Adaptive scaling based on recent residuals
            window_size = 5
            recent_residuals = residuals[-window_size:] if len(residuals) > window_size else residuals
            recent_x = [r[0] for r in recent_residuals]
            recent_y = [r[1] for r in recent_residuals]

            recent_dists = [np.sqrt(x**2 + y**2) for x, y in zip(recent_x, recent_y)]
            recent_max_dist = max(recent_dists) if recent_dists else 0

            min_scale = 1.0
            axis_limit = max(min_scale, recent_max_dist * 1.5)

            self.ax_residuals.set_xlim(-axis_limit, axis_limit)
            self.ax_residuals.set_ylim(-axis_limit, axis_limit)

            self.ax_residuals.grid(True, alpha=0.3)
            self.ax_residuals.set_title(f"Residual: {np.sqrt(x_res[-1]**2 + y_res[-1]**2):.2f} px", fontsize=7)
            self.ax_residuals.set_xlabel("X Residual (px)")
            self.ax_residuals.set_ylabel("Y Residual (px)")
        

        plt.draw()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
        gc.collect()
 
    
    def close(self):
        """Close the plot."""
        plt.close(self.fig)
        plt.close('all')  # Close any other orphaned figures
        gc.collect()      # Force garbage collection
        
    def save(self, filename):
        """Save the current figure to a file."""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')