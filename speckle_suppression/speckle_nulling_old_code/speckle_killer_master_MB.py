import os
import sys
import numpy as np

from configobj import ConfigObj
import matplotlib.pyplot as plt
import astropy.io.fits as pf
import sn_math as snm
import sn_preprocessing as pre
import sn_sims as sns
import sn_filehandling as flh
import scipy.ndimage as sciim
# import sn_hardware as hardware
import sn_processing as pro
from detect_speckles import create_speckle_aperture, get_speckle_photometry

from plot_explorer import Plotter
import dm_functions as DM
import time
import detect_speckles
import dm_registration as DMR
import compute_contrastcurve as cc
from copy import deepcopy
from scipy.spatial.distance import euclidean
import qacits_routines as qac
import ipdb

class Logger(object):
    def __init__(self, fname):
        self.terminal = sys.stdout
        self.log = open(fname, "a+")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()


class speckle:
    def __init__(self, image,xp, yp, config):
        self.imparams = config['IM_PARAMS']
        self.abc       = config['INTENSITY_CAL']['abc']
        self.xcentroid = xp
        self.ycentroid = yp
        self.kvec      = DM.convert_pixels_kvecs(self.xcentroid,
                                                 self.ycentroid,
                                                 **self.imparams)
        self.kvecx     = self.kvec[0]
        self.kvecy     = self.kvec[1]
        self.krad      = np.linalg.norm((self.kvecx, self.kvecy))
        self.aperture  = detect_speckles.create_speckle_aperture(
                image, self.xcentroid, self.ycentroid, config['INTENSITY_CAL']['aperture_radius'])
        self.exclusionzone  = detect_speckles.create_speckle_aperture(
                image, self.xcentroid, self.ycentroid, config['NULLING']['exclusionzone'])
        if config['DETECTION']['doublesided']:
            print(( "using double sided dark hole"))
            #Edit to take into account aperture on other side
            self.aperture = self.aperture + detect_speckles.create_speckle_aperture(
                    image, 2*self.imparams['centerx']-self.xcentroid,
                           2*self.imparams['centery']-self.ycentroid,
                           config['INTENSITY_CAL']['aperture_radius'])
            #Edit to take into account aperture on other side
            self.exclusionzone  = self.exclusionzone + detect_speckles.create_speckle_aperture(
                    image, 2*self.imparams['centerx']-self.xcentroid,
                           2*self.imparams['centery']-self.ycentroid,
                           config['NULLING']['exclusionzone'])

        self.intensity = detect_speckles.get_speckle_photometry(image, self.aperture)
        #self.finalintensity = None
        self.phase_intensities = [None, None, None, None]
        self.phases = config['NULLING']['phases']
        self.null_phase = None

        self.null_gain = None
        self.gains= config['NULLING']['amplitudegains']
        self.gain_intensities = [None, None, None, None]

    def recompute_intensity(self, phaseimage):
        return detect_speckles.get_speckle_photometry(phaseimage, self.aperture)

    def generate_flatmap(self, phase):
        """generates flatmap with a certain phase for this speckle"""
        s_amp = DM.amplitudemodel(self.intensity, self.krad, **self.abc)
        #print('s_amp: ', s_amp, 'self.krad: ', self.krad, 'phase: ', phase, ' xy =',self.xcentroid, self.ycentroid	)
        return DM.make_speckle_kxy(self.kvecx, self.kvecy, s_amp, phase)

    def compute_null_phase(self):
        A, B, C, D = self.phase_intensities
        #self.null_phase =  -1*np.arctan2((D-B),(A-C))+np.pi
        self.null_phase = np.arctan2((B-D), (A-C)) - np.pi
        return self.null_phase

    def compute_null_gain(self):
        strictly_increasing = all(x<y for x, y in zip(self.gain_intensities, self.gain_intensities[1:]))
        strictly_decreasing = all(x<y for x, y in zip(self.gain_intensities, self.gain_intensities[1:]))
        bestgain = self.gains[self.gain_intensities.index(min(self.gain_intensities))]
        if strictly_increasing:
            self.null_gain = bestgain
        elif strictly_decreasing:
            self.null_gain = bestgain
        else:
            #fit a decreasing parabola
            a, b, c = np.polyfit(self.gains, self.gain_intensities, deg=2)
            if a<1:
                print("WARNING: got an upward sloping parabola! Using best result.")
                self.null_gain = bestgain
            else:
                self.null_gain =-1.0*b/(2*a)
                if self.null_gain > max(self.gains):
                    print("WARNING: computed gain is greater than ",
                            max(self.gains),
                            ", using best result")
                    self.null_gain = bestgain
                elif self.null_gain < min(self.gains):
                    print("WARNING: computed gain is less than ",
                            min(self.gains),
                            ", using best result")
                    self.null_gain = bestgain
                else:
                    pass
        print("NULL GAIN IS: ", self.null_gain)
        return self.null_gain


def identify_bright_points(image, controlregion=None,
                           size=None):
    """
    Runs a maximum filter over an image and returns coords
    of bright points in the control region.
    Control region should be an array of 1's and 0's
    size is the nxn size of the maximum filter"""
    if controlregion is not None:
        image = image*controlregion

    max_filt = sciim.filters.maximum_filter(image, size=size)

    pts_of_interest = (max_filt == image)
    pts_of_interest_in_region = pts_of_interest*controlregion
    iindex, jindex = np.nonzero((max_filt*controlregion == image*controlregion)*controlregion)
    intensities = np.zeros(iindex.shape)
    for i in range(len(intensities)):
        intensities[i] = image[iindex[i], jindex[i]]
    order = np.argsort(intensities)[::-1]
    sorted_i = iindex[order]
    sorted_j = jindex[order]
    xyofinterest = [p[::-1] for p in zip(sorted_i, sorted_j)]
    return xyofinterest

def get_waffle_brightness(kvecr,im,centerx=None, centery=None, angle = None,lambdaoverd= None ):
    # xy0 = DM.convert_kvecs_pixels(kvecr, 0,centerx,centery,angle,lambdaoverd)
    # xy1 = DM.convert_kvecs_pixels(-kvecr, 0,centerx,centery,angle,lambdaoverd)
    # xy2 = DM.convert_kvecs_pixels(0,kvecr,centerx,centery,angle,lambdaoverd)
    # xy3 = DM.convert_kvecs_pixels(0,-kvecr,centerx,centery,angle,lambdaoverd)
    xy0 = DM.convert_kvecs_pixels(kvecr, kvecr,centerx,centery,angle,lambdaoverd)
    xy1 = DM.convert_kvecs_pixels(-kvecr, kvecr,centerx,centery,angle,lambdaoverd)
    xy2 = DM.convert_kvecs_pixels(-kvecr,-kvecr,centerx,centery,angle,lambdaoverd)
    xy3 = DM.convert_kvecs_pixels(-kvecr,-kvecr,centerx,centery,angle,lambdaoverd)
    xypixels = [xy0, xy1, xy2, xy3]
    ximcoords, yimcoords = np.meshgrid(np.arange(im.shape[0]),np.arange(im.shape[1]))
    meankvec = 0
    meanphotom = 0
    for idx, xy in enumerate(xypixels):
        subim = pre.subimage(im, xy, window = 10)
        subx  = pre.subimage(ximcoords, xy, window = 10)
        suby  = pre.subimage(yimcoords, xy, window = 10)
        gauss_params = snm.fitgaussian(subim, subx, suby)
        spotx, spoty = gauss_params[1], gauss_params[2]
        aperture = create_speckle_aperture(im,spotx, spoty,ap_rad)
        photometry = get_speckle_photometry(im, aperture)
        meanphotom = (meanphotom*float(idx)/float(idx+1) +
                             photometry/float(idx+1))
    return meanphotom

def filterpoints(pointslist, rad=6.0, max=5, cx=None, cy=None):
    if len(pointslist)<max:
        max = len(pointslist)
    plist = pointslist[:]
    passed = []
    for item in plist:
        passed.append(item)
        plist[:] = [x for x in plist
                        if (euclidean(item, x)>rad and
                        euclidean(2*np.array((cx, cy))-np.array(item), x)>rad) or
                        x in passed]
        print(" ")
    if len(plist)>max:
        plist = plist[0:max]
    return plist

def generate_phase_nullmap(speckleslist, gain):
    nullmap = 0
    for speck in speckleslist:
        null_phase=speck.compute_null_phase()
        nullmap = nullmap+gain*speck.generate_flatmap(null_phase)

    return nullmap

def generate_super_nullmap(speckleslist):
    nullmap = 0
    for speck in speckleslist:
        null_phase=speck.compute_null_phase()
        nullmap = nullmap+speck.null_gain*speck.generate_flatmap(null_phase)

    return nullmap

def printstats(cleaned_image, speckleslist):
    def fmt(x):
        if x is not None:
            return '%.2f'%x
        else:
            return ''
    percent_improvements = []
    total_ints = 0
    null_gains = []
    c_percent_improv = []
    controlled_nms= 0
    outputstr = '\n'
    for speck in speckleslist:
        orig = speck.intensity
        final = speck.recompute_intensity(cleaned_image)
        #print(speck.xcentroid, speck.ycentroid)

        perc_imp = 100.0*(orig-final)/orig
        percent_improvements.append(perc_imp)
        total_int = final-orig
        total_ints += total_int
        null_gains.append(speck.null_gain)
        sp_str = ( ("Pos: "+fmt(speck.xcentroid)+', '+fmt(speck.ycentroid)).ljust(20)+
                   ("Orig int.: "+fmt(orig)).ljust(20)+
                   ("Final int.: " + fmt(final)).ljust(20)+
                   ("Null Gain:" + fmt(speck.null_gain)).ljust(20)+
                   ("% improv: "+fmt(perc_imp)).ljust(20)
                   )
        outputstr += sp_str+'\n'
    return outputstr

def check_AO(soft_ini=None, soft_spec=None,
                image=None, AO = None):
    return (128+np.random.normal(), 128+np.random.normal), 200+np.random.normal()*10

class Imagestats:
    def __init__(self):
        self.maxfluxes = []
        self.meanfluxes = []
        self.totalfluxes = []
        self.rmsfluxes = []
        self.itcounter = None
    def update(self, data, controlregion = None):
        if controlregion is None:
            controlregion = np.ones(data.shape)
        assert np.shape(controlregion) == np.shape(data)
        data_in_region = data[controlregion>0]
        self.meanfluxes.append(np.mean(data_in_region))
        self.maxfluxes.append(np.max(data_in_region))
        self.totalfluxes.append(np.sum(data_in_region))
        self.rmsfluxes.append(np.std(data_in_region))
        self.itcounter = list(range(len(self.meanfluxes)))
        return
    def print_latest_imstats(self):
        labels = ['mean','max', 'total', 'rms']
        items = [self.maxfluxes, self.meanfluxes, self.totalfluxes,
                 self.rmsfluxes]
        print("Iteration: ",self.itcounter[-1])
        for label, item in zip(labels, items):
            print("Flux "+label+'%.2f'%item[-1])
        return

if  __name__ == "__main__":
    # Initialized config and spec file names
    soft_ini  = 'Config/SN_Software.ini'
    soft_spec = 'Config/SN_Software.spec'
    hard_ini  = 'Config/SN_Hardware.ini'
    hard_spec = 'Config/SN_Hardware.spec'

    # Read the config files and check if satisfied spec files requierements.
    config = flh.validate_configfile(soft_ini, soft_spec)
    configfilename = soft_ini
    # Hardware Connection
    print('')
    print('############################################################')
    print('#################### Hardware Connection ###################')
    print('############################################################')
    print('')

    # Instancie AO system and Detector selected
    # Here we run the simulator

    KeckSim = sns.FakeCoronagraph()
    KeckSim.make_DM()
    KeckSim.make_TTM()
    x0 = np.array([0.0,0.0])
    KeckSim.set_TTM(x0)
    cal_waffle = DM.make_speckle_kxy(-10., 10., 0.005, 0) + DM.make_speckle_kxy(10., 10., 0.005, 0)
    KeckSim.set_dm_shape(cal_waffle)
    KeckSim.make_aberration(0.1)
    # AOsystem = snh.KECK2AO('KECKAO', hard_ini, hard_spec)
    # Detector = snh.NIRC2(  'NIRC2' , hard_ini, hard_spec)

    update_bgd = False

    im_params= config['IM_PARAMS']
    null_params = config['NULLING']
    abc = config['INTENSITY_CAL']['abc']

    intconf = config['INTENSITY_CAL']
    ap_rad = intconf['aperture_radius']

    bgds = flh.setup_bgd_dict(config)
    #FIX this, controlregion is in local dir.
    controlregion = pf.open(config['CONTROLREGION']['filename'])[0].data
    N_iterations = config['NULLING']['N_iterations']

    min_contrastcurve = cc.contrastcurve_simple(bgds['bkgd'],
                                     cx = config['IM_PARAMS']['centerx'],
                                     cy = config['IM_PARAMS']['centery'],
                                     region = controlregion,
                                     robust = True, fwhm = 6.0,
                                     maxrad = 50)
    #NEED TO CHECK -- THIS MIGHT BE HALF THE MIN BACKGROUND SINCE BGD IS SUBTRACTED
    #Setup
    initial_dm_shape = KeckSim.get_dm_shape()
    defaultim = np.ones(bgds['masterflat'].shape)

    Imstats = Imagestats()

    Plotter = Plotter(config=config, defaultim=defaultim,
                      controlregion=controlregion)

    Datasaver = flh.DataSaver(defaultim=defaultim, config=config)
    Datasaver.initialize()
    cubeoutputdir = Datasaver.outputdir
    tstamp = Datasaver.tstamp
    sys.stdout = Logger(os.path.join(cubeoutputdir, 'logfile.txt'))
    print("CUBE OUTPUT DIR", cubeoutputdir)
    print("making resultcubes 1")
    result_imagecube =  flh.Output_imagecube(
                           N_iterations, defaultim,
                           filepath = os.path.join(cubeoutputdir,
                                        'test_'+ tstamp+'.fits'),
                           comment = ' ' ,
                           configfile = configfilename)

    print("making resultcubes 2")
    clean_imagecube=  flh.Output_imagecube(
                           N_iterations, defaultim,
                           filepath = os.path.join(cubeoutputdir,
                                        'test_clean_'+ tstamp+'.fits'),
                           comment = ' ',
                           configfile = configfilename)

    ####MAIN LOOP STARTS HERE#####
    print("BEGINNING NULLING LOOP" )

    recompute_center = config['RECENTER']['recenter']
    #NEEDTOFIX MAKE SURE 15 and 20 ARE SAFE AND DON'T OVERRRUN IMAGE
    bgd_pix = pro.annulus(np.zeros(defaultim.shape),
                           config['IM_PARAMS']['centerx'],
                           config['IM_PARAMS']['centery'],
                           int(15*config['IM_PARAMS']['lambdaoverd']),
                           int(20*config['IM_PARAMS']['lambdaoverd']))
    wherebgd = np.where(bgd_pix)

    for iteration in range(N_iterations):
        # current_dm_solution = AOsystem.get_dm_shape()
        current_dm_solution = KeckSim.get_dm_shape()
        Datasaver.write_dm_shape(current_dm_solution, it=iteration, name='start')
        print("Taking image of speckle field")
        raw_im = KeckSim.take_image()
        if update_bgd == True:
            updatelev = np.median(raw_im[wherebgd])/np.median(bgds['bkgd'][wherebgd])
            bgds['bkgd'] = bgds['bkgd']*updatelev
            print("scaling background by ", updatelev)
        cleaned_image = pre.equalize_image(raw_im, **bgds)
        Datasaver.write_image(raw=raw_im, clean=cleaned_image, it=iteration,
                             offset=None, intensity=None)

        Imstats.update(cleaned_image, controlregion=controlregion)
        Imstats.print_latest_imstats()
        trou = get_waffle_brightness(10,cleaned_image,**im_params)
        print("MEAN FLUX OF WAFFLES IS ", trou)

        if config['NULLING']['scale_intensity']:
            innerim = cleaned_image*innerctrlann
            basemedianflux = np.median(innerim[innerim>0])
            print("Base median flux in inner region is ", basemedianflux)

        Plotter.plot(Imstats.itcounter ,Imstats.rmsfluxes, ax=Plotter.ax3)

        pixrad, clevel = cc.contrastcurve_simple(cleaned_image,
                                   cx = config['IM_PARAMS']['centerx'],
                                   cy = config['IM_PARAMS']['centery'],
                                    region = controlregion,
                                    #FIX FWHM THING
                                    robust = True, fwhm = 6.0,
                                    maxrad = 50)
        Plotter.plot(pixrad, clevel/float(
                        config['NULLING']['referenceval']),
                     ax=Plotter.ax4)

        Plotter.ax1.set_title('Iteration: '+str(iteration))
        Plotter.update_main_image(cleaned_image)
        # Check if there is an improvement
        if iteration >0:
            ##FIX THIS
            config = flh.validate_configfile(soft_ini, soft_spec)
            stats = printstats(cleaned_image, speckleslist)
            print(stats)
            Datasaver.save_textfile(stats, it=iteration,name='specklestats.txt')
            if recompute_center == True:
                print("Recentering")
        #NEED TO FIX
        #prompt user
        #ans = input('Do you want to run a speckle nulling iteration[Y/N]?')
        ans = True
        if ans == 'N':
           flatmapans = input('Do you want to reload the'+
                                  'initial DM shape you started with[Y/N]?')
           if flatmapans == 'Y':
               print("Reloading initial DM shape")
               status = AOsystem.set_dm_shape(initial_dm_shape)
           break

        print('Iteration '+str(iteration)+
              ' total_intensity: '+str(Imstats.totalfluxes[-1]))
        #return a list of points
        print("computing interesting bright spots")

        #note indices and coordinates are reversed
        xyofinterest= identify_bright_points(cleaned_image, controlregion,
                                            size=config['DETECTION']['window'])

        print("computed ",str(len(xyofinterest)), " bright spots")

        fps = filterpoints(xyofinterest,
                           max = config['DETECTION']['max_speckles'],
                           rad=config['NULLING']['exclusionzone'],
                           cx = config['IM_PARAMS']['centerx'],
                           cy = config['IM_PARAMS']['centery'])
        print("creating speckle objects")
        speckleslist = [speckle(cleaned_image, xy[0], xy[1], config) for xy in fps]

        for idx, phase in enumerate(null_params['phases']):
            print("Phase ", phase)
            phaseflat = 0
            allspeck_aps= 0
            #put in sine waves at speckle locations
            for speck in speckleslist:
                #XXX
                phaseflat = phaseflat + speck.generate_flatmap(phase)
                allspeck_aps = allspeck_aps + speck.aperture
            # status = AOsystem.set_dm_shape(current_dm_solution +phaseflat)
            status = KeckSim.set_dm_shape(current_dm_solution +phaseflat)

            # phaseim = Detector.take_image()
            #Take an image
            raw_phaseim = KeckSim.take_image()
            if update_bgd == True:
                updatelev = np.median(raw_im[wherebgd])/np.median(bgds['bkgd'][wherebgd])
                bgds['bkgd'] = bgds['bkgd']*updatelev
                print("scaling background by ", updatelev)
            phaseim = pre.equalize_image(raw_phaseim, **bgds)
            if recompute_center == True:
                print("Recentering")
            Datasaver.write_phaseim(clean=phaseim, raw=raw_phaseim, it=iteration,
                                    phase_it=idx, offset=None)

            if config['NULLING']['scale_intensity']:
                phaseimmedianflux = np.median((phaseim*innerctrlann)[innerctrlann>0])
                print("median flux in region is ", phaseimmedianflux)
                factor = basemedianflux/phaseimmedianflux
                print("scaling image by ", factor)
                phaseim = phaseim*factor
            Plotter.update_speckle_image(phaseim,
                                         speckle_aps=allspeck_aps,
                                         title='Phase: '+'%.2f'%phase)
            Datasaver.write_apertureim(speckle_aps=allspeck_aps, it=iteration)
            print("recomputing intensities")
            for speck in speckleslist:
                phase_int = speck.recompute_intensity(phaseim)
                speck.phase_intensities[idx] = phase_int

        nullmap= generate_phase_nullmap(speckleslist,
                                        config['NULLING']['default_flatmap_gain'])
        # AOsystem.set_dm_shape(current_dm_solution)
        KeckSim.set_dm_shape(current_dm_solution+nullmap)
        Datasaver.write_dm_shape(nullmap, it=iteration, name='offset')
        Datasaver.write_dm_shape(current_dm_solution+nullmap, it=iteration, name='end')
    print("Max iterations reached.  Exiting gracefully.  Run again if you like")
