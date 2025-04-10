import matplotlib.pyplot as plt
import sn_filehandling as flh
import numpy as np
import astropy.io.fits as pf
import compute_contrastcurve as cc
import ipdb
import sys
import scipy.ndimage as sciim

class Plotter:
    def __init__(self, config=None,
                       defaultim=None,
                       controlregion=None):
        self.config = config
        self.xlims, self.ylims= self.get_plot_bounds(
                                     self.config['CONTROLREGION']['verticesx'],
                                     self.config['CONTROLREGION']['verticesy'])
        self.n_iter = self.config['NULLING']['N_iterations']
        self.ims_per_iter = len(config['NULLING']['phases'])+1
        #Create plot
        plt.ion()
        fig = plt.figure(figsize = (8, 12))
        fig.subplots_adjust(hspace=0) # height spaces
        fig.subplots_adjust(wspace=0) # width spaces
        self.ax1 = plt.subplot2grid((6,4),(0, 0), rowspan=2, colspan=2)
        self.ax2 = plt.subplot2grid((6,4),(0, 2), rowspan=2, colspan=2)
        self.ax3 = plt.subplot2grid((6,4),(2, 0), rowspan=2, colspan=2)
        self.ax4 = plt.subplot2grid((6,4),(2, 2), rowspan=2, colspan=2)
        self.ax5 = plt.subplot2grid((6,4),(4, 0), rowspan=2, colspan=2)
        self.ax6 = plt.subplot2grid((6,4),(4, 2), rowspan=2, colspan=2)
        self.ax1.set_title('Image')
        self.ax2.set_title('Control region')
        self.ax3.set_title('RMS in region')
        self.ax4.set_title('Raw 1s contrast: ref '+
                        str(self.config['NULLING']['referenceval']))
        self.ax5.set_title('Satellite Centroid')
        self.ax6.set_title('Satellite Intensity')
        for ax in [self.ax1, self.ax2]:
            ax.set_xlim(self.xlims)
            ax.set_ylim(self.ylims)
            ax.axis('off')
        self.controlregion = controlregion
        self.border = self.get_border(self.controlregion)
        self.w1 = self.imshow(defaultim, ax=self.ax1)
        self.w2 = self.imshow(self.controlregion*defaultim, ax=self.ax2)
        ##RMS in region plot
        self.w3 = self.ax3.plot(np.arange(self.n_iter),np.repeat(0, self.n_iter), 'k.')
        #self.ax3.set_xlim(0, self.n_iter)
        ##Contrast plot
        #self.w4 = self.ax4.plot(np.arange(self.n_iter), np.repeat(min_rms, self.n_iter), 'k.')

        self.w5, = self.ax5.plot(config['RECENTER']['setpointx'],
                                 config['RECENTER']['setpointy'],
                                 marker="P", markersize=10, linewidth=11, color='red')
        self.ax5.set_xlim(config['RECENTER']['setpointx']-5,config['RECENTER']['setpointx']+5)
        self.ax5.set_ylim(config['RECENTER']['setpointy']-5,config['RECENTER']['setpointy']+5)
        self.w6, = self.ax6.plot(np.arange(self.n_iter*self.ims_per_iter),
                                 np.random.random(self.n_iter*self.ims_per_iter))

        plt.show()
        return

    def scale(self, data):
        """scales an image for plotting"""
        return np.log(np.abs(data))

    def imshow(self, data, ax=None):
        out = ax.imshow(self.scale(data), origin='lower', interpolation='nearest')
        return out

    def plot(self, xdata, ydata, ax=None):
        ax.plot(xdata, ydata)
        plt.draw()
        plt.pause(0.002)
        pass

    def update_main_image(self, data, title=None):
        self.w1.set_data(self.scale(data)+self.border)
        self.w1.autoscale()
        if title is not None:
            self.ax1.set_title(title)
        plt.draw()
        plt.pause(0.002)
        return

    def update_setpoint_image(self, xdata, ydata):
        #self.w5.clear()
        self.ax5.plot(xdata, ydata, alpha = 0.5)
        self.ax5.plot(xdata[-1], ydata[-1],
                         color='magenta', marker='o', alpha = 0.8)
        plt.draw()
        plt.pause(0.002)
        return

    def update_speckle_image(self, data, title=None,
                             speckle_aps=None):
        im = self.scale(data*self.controlregion)
        if title is not None:
            self.ax2.set_title(title)
        if speckle_aps is not None:
            im = im+self.get_border(speckle_aps)
        self.w2.set_data(im)
        self.w2.autoscale()
        plt.draw()
        plt.pause(0.002)
        return

    def get_border(self, controlregion):
        """Given a control region, compute the border using
        a laplace filter"""
        border = np.abs(sciim.filters.laplace(controlregion))
        border = border*1.0
        try:
            border[border>0] = np.nan
        except:
            ipdb.set_trace()
        return border

    def get_plot_bounds(self, vertsx, vertsy):
        """
        get image bounds from annulus center, inner, outer coords
        verts in the form of
        (xcenter, xinner, xouter),
        (ycenter, yinner, youter)"""
        anncentx, anncenty = vertsx[0], vertsy[0]
        annrad = np.sqrt( (vertsx[0]-vertsx[2])**2+
                          (vertsy[0]-vertsy[2])**2)
        xlims = anncentx-annrad, anncentx+annrad
        ylims = anncenty-annrad, anncenty+annrad
        return xlims, ylims


if __name__ == "__main__":
    soft_ini  = 'Config/SN_Software.ini'
    soft_spec = 'Config/SN_Software.spec'
    config = flh.validate_configfile(soft_ini, soft_spec)
    bgds = flh.setup_bgd_dict(config)
    defaultim = np.ones(bgds['masterflat'].shape)
    controlregion = pf.open(config['CONTROLREGION']['filename'])[0].data
    vertsx = config['CONTROLREGION']['verticesx']
    vertsy = config['CONTROLREGION']['verticesy']

    Plotter = Plotter(config=config, defaultim=defaultim, controlregion=controlregion)
    #Plotter.imshow(defaultim, ax=Plotter.ax1)
    #sys.exit(1)
    #anncentx, anncenty = vertsx[0], vertsy[0]
    #annrad = np.sqrt( (vertsx[0]-vertsx[2])**2+
    #                  (vertsy[0]-vertsy[2])**2)

    #N_iterations = config['NULLING']['N_iterations']
    #min_rms = (np.std(bgds['bkgd'][np.where(controlregion>0)])/
    #                 config['NULLING']['referenceval'])

    #min_contrastcurve = cc.contrastcurve_simple(bgds['bkgd'],
    #                                 cx = config['IM_PARAMS']['centerx'],
    #                                 cy = config['IM_PARAMS']['centery'],
    #                                 region = controlregion,
    #                                 robust = True, fwhm = 6.0,
    #                                 maxrad = 50)

    #fig = plt.figure(figsize = (10, 10))
    #ax1 =plt.subplot2grid((4,4),(0, 0), rowspan =2, colspan = 2)
    #ax2 = plt.subplot2grid((4,4),(0, 2), rowspan =2, colspan = 2)
    #ax3 =plt.subplot2grid((4,4),(2, 0), rowspan =3, colspan = 2)
    #ax4 =plt.subplot2grid((4,4),(2, 2), rowspan =3, colspan = 2)

    #title = fig.suptitle('Speckle destruction')
    #ax1.set_title('Image')
    #ax2.set_title('Control region')
    #ax3.set_title('RMS in region')
    #ax4.set_title('Raw 1s contrast: ref '+
    #                str(config['NULLING']['referenceval']))

    #w1 = ax1.imshow(np.log(np.abs(defaultim)), origin='lower', interpolation = 'nearest')
    #ax1.set_xlim(anncentx-annrad, anncentx+annrad)
    #ax1.set_ylim(anncenty-annrad, anncenty+annrad)
    #w2 = ax2.imshow(np.log(np.abs(controlregion*defaultim)), origin='lower', interpolation = 'nearest')
    #ax2.set_xlim(anncentx-annrad, anncentx+annrad)
    #ax2.set_ylim(anncenty-annrad, anncenty+annrad)

    #w3 = ax3.plot(np.arange(N_iterations),np.repeat(min_rms, N_iterations), 'k.')
    #ax3.set_xlim(0, N_iterations)

    #w4 = ax4.plot(min_contrastcurve[0], min_contrastcurve[1], 'k.')
