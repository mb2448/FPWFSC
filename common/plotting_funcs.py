import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from hcipy import imshow_field
import numpy as np

plt.rcParams.update({
    'font.size': 8,  # Base font size
    'axes.labelsize': 8,  # Axis labels
    'axes.titlesize': 9,  # Subplot titles
    'xtick.labelsize': 7,  # X-axis tick labels
    'ytick.labelsize': 7,  # Y-axis tick labels
    'legend.fontsize': 7,  # Legend text
    'figure.titlesize': 10  # Figure title
})

class LivePlotter:
    def __init__(self):
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(4,4))
        self.setup_subplots()
        
    def setup_subplots(self):
        # Create the same subplot layout
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)
        
        # Initialize plots
        self.psf_img = self.ax1.imshow(np.zeros((10,10)))
        self.wf_img = self.ax2.imshow(np.zeros((10,10)), cmap='bwr')
        self.strehl_line, = self.ax3.plot([], [])
        self.var_line, = self.ax4.plot([], [])
        
        # Add colorbars and labels
        self.fig.colorbar(self.psf_img, ax=self.ax1, fraction=0.046, pad=0.04)
        self.fig.colorbar(self.wf_img, ax=self.ax2, fraction=0.046, pad=0.04)
        
        self.ax3.set_ylabel('Strehl')
        self.ax3.set_ylim([0, 1.1])
        self.ax4.set_ylabel('VAR')
        self.fig.tight_layout()
        
    def update(self, Niter, data, pupil_wf, aperture, SRA, VAR):
        i = np.sum(~np.isnan(SRA))
        
        # Update PSF
        self.ax1.clear()
        imshow_field(np.log10(np.abs(data) / data.max() + 1E-8), vmin=-5, vmax=0, ax=self.ax1)
        self.ax1.set_title(f'iteration {i+1}/{Niter}')
        
        # Update residuals
        self.ax2.clear()
        max_res = np.max(np.abs(pupil_wf * (aperture > 0)))
        imshow_field(pupil_wf * (aperture > 0), cmap='bwr', vmin=-max_res, vmax=max_res, ax=self.ax2)
        self.ax2.set_title('current residuals')
        
        # Update Strehl
        self.ax3.clear()
        self.ax3.plot(np.arange(Niter), SRA)
        self.ax3.set_title(f'Current Strehl = {SRA[i-1]:.3f}')
        self.ax3.set_xlabel('iteration')
        self.ax3.set_ylabel('Strehl')
        self.ax3.set_ylim([0,1.1])
        self.ax3.grid(True, axis='y')
        self.ax3.set_xlim([0,Niter])
        
        # Update VAR
        self.ax4.clear()
        self.ax4.plot(np.arange(Niter), VAR)
        self.ax4.set_title(f'Current VAR = {VAR[i-1]:.3f}')
        self.ax4.set_xlabel('iteration')
        self.ax4.set_ylabel('VAR')
        self.ax4.set_xlim([0,Niter])
        self.ax4.set_yscale('log')
        self.ax4.set_ylim([1e-4, 1])
        self.ax4.grid(True, axis='y')
        
        plt.draw()
        #plt.pause(0.00001)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()