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
# import sn_hardware as hardware
import sn_processing as pro

import dm_functions as DM
import time
import detect_speckles
import dm_registration as DMR
import compute_contrastcurve as cc
from copy import deepcopy
from scipy.spatial.distance import euclidean
import scipy.ndimage as sciim
import qacits_routines as qac

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

class output_imagecube:
    def __init__(self, n, defaultim, filepath = None, comment = None, configfile = None):
        self.size_x = defaultim.shape[0]
        self.size_y = defaultim.shape[1]
        self.cube = np.zeros( (n,self.size_x, self.size_y))
        self.textstring = (comment + '\n\n\n'+self.config_to_string(configfile))
        flh.writeout(self.cube, outputfile = filepath, 
                            comment =comment)
        self.i = 0 
        self.filepath = filepath

        with open(filepath+'.txt', 'w') as f:
            f.write(self.textstring)
    
    def config_to_string(self, configfile):
        stringy = ''
        with open(configfile) as f:
            for line in f:
                stringy = stringy+line
        return stringy

    def update(self, array ):
        self.cube[self.i, :,:] = array
        self.i = self.i+1
        flh.writeout(self.cube, outputfile = self.filepath)
                            
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
        print('s_amp: ', s_amp, 'self.krad: ', self.krad, 'phase: ', phase, ' xy =',self.xcentroid, self.ycentroid	)
        return DM.make_speckle_kxy(self.kvecx, self.kvecy, s_amp, phase)
    
    def compute_null_phase(self):
        A, B, C, D = self.phase_intensities 
        #self.null_phase =  -1*np.arctan2((D-B),(A-C))+np.pi
        self.null_phase = np.arctan2((B-D), (A-C)) - np.pi 
        return self.null_phase
    
    def compute_null_gain(self):
        #L = self.gain_intensities
        
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


def identify_bright_points(image):
    """WARNING: indexes, NOT coordinates"""
    ##NEED TO FIX.  CONTROL REGION GLOBAL VARIABLE 
    max_filt = sciim.filters.maximum_filter(image, size= 10)
    
    pts_of_interest = (max_filt == image)
    pts_of_interest_in_region = pts_of_interest*controlregion
    iindex, jindex = np.nonzero((max_filt*controlregion == image*controlregion)*controlregion)
    intensities = np.zeros(iindex.shape)
    for i in range(len(intensities)):
        intensities[i] = image[iindex[i], jindex[i]]
    order = np.argsort(intensities)[::-1]
    sorted_i = iindex[order]
    sorted_j = jindex[order]
    return zip(sorted_i, sorted_j)

def filterspeckles(specklelist, max=5):
    copylist= deepcopy(specklelist)
    #FIRST ELEMENT ALWAYS SELECTED
    sum = copylist[0].exclusionzone*1.0
    returnlist = []
    returnlist.append(copylist[0])
    if (max == 1):
        return returnlist
    if max >1:
        i=1

        while len(returnlist)<max:
            if i>=len(copylist):
                break
            #print("max: ",max," i: ",i)
            sum_tmp = sum+copylist[i].exclusionzone*1.0 
            #print("test")
            #print(i, (sum_tmp>1).any())
            if (sum_tmp>1).any():
                i=i+1
                sum_tmp = sum
            else:
                returnlist.append(copylist[i])
                sum = sum_tmp
                i = i+1
        return returnlist


def filterpoints(pointslist, rad=6.0, max=5, cx=None, cy=None):
    plist = pointslist[:]
    print((plist))
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
    
            
def fastplot(specklelist):
    fig2 = plt.figure(figsize=(4,4))
    ax2 = fig2.add_subplot(111)
    for speck in specklelist:
        ax2.plot(speck.phases, speck.phase_intensities)
        ax2.set_title(( str(speck.xcentroid)+', '+
                        str(speck.ycentroid)))
        ax2.axvline(speck.null_phase)
        print(speck.null_phase)
        plt.draw()
        plt.pause(0.1)
        plt.cla()
    plt.close()
    pass

def printstats(cleaned_image, speckleslist):
    percent_improvements = []
    total_ints = 0
    null_gains = []
    c_percent_improv = []
    controlled_nms= 0
    for speck in speckleslist:
        orig = speck.intensity
        final = speck.recompute_intensity(cleaned_image)
        #print(speck.xcentroid, speck.ycentroid)
   
        perc_imp = 100.0*(final-orig)/orig
        percent_improvements.append(perc_imp)
        total_int = final-orig
        total_ints += total_int
        null_gains.append(speck.null_gain)
        print( "Position: "+str(speck.xcentroid)+", "+str(speck.ycentroid)+" " + 
                "Orig intensity: "+str(int(speck.intensity))+" "+
               "Final intensity: " + str(int(final))+"  "+
               'Null Gain:' + str(speck.null_gain)+"  "+
               "Percent improv: "+str(perc_imp))
        if speck.null_gain is not None:
            #NEED TO FIX WTF IS THIS
            if speck.null_gain>0:
                c_percent_improv.append(perc_imp)
                controlled_nms+= total_int
    print("\nTotal amplitude change "+str(total_ints)+
           "\nNonzero gain amplitude change: "+str(controlled_nms)+
           "\nMean percent improvement: "+str(np.mean(percent_improvements))+
           "\nMean nonzero gain percent improvement: "+str(np.mean(c_percent_improv)))

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
    # speckle_x0 = DM.make_speckle_kxy(4.,0., 0.1, 0)
    # abb0 = speckle_x0
    # KeckSim.set_dm_shape(abb0)
    cal_waffle = DM.make_speckle_kxy(-10., 10., 0.005, 0) + DM.make_speckle_kxy(10., 10., 0.005, 0)
    KeckSim.set_dm_shape(cal_waffle)
    KeckSim.make_aberration(0.1)
    # AOsystem = snh.KECK2AO('KECKAO', hard_ini, hard_spec)
    # Detector = snh.NIRC2(  'NIRC2' , hard_ini, hard_spec)

    update_bgd = False
    
    im_params= config['IM_PARAMS']
    null_params = config['NULLING']
    abc = config['INTENSITY_CAL']['abc']

    bgds = flh.setup_bgd_dict(config)
    controlregion = pf.open(config['CONTROLREGION']['filename'])[0].data
    N_iterations = config['NULLING']['N_iterations']

    min_contrastcurve = cc.contrastcurve_simple(bgds['bkgd'],
                                     cx = config['IM_PARAMS']['centerx'],
                                     cy = config['IM_PARAMS']['centery'],
                                     region = controlregion,
                                     robust = True, fwhm = 6.0,          
                                     maxrad = 50)
    #NEED TO CHECK -- THIS MIGHT BE HALF THE MIN BACKGROUND SINCE BGD IS SUBTRACTED
    min_rms = (np.std(bgds['bkgd'][np.where(controlregion>0)])/
                     config['NULLING']['referenceval'])
    #Setup
    initial_dm_shape = KeckSim.get_dm_shape()
    defaultim = np.ones(bgds['masterflat'].shape)

    vertsx = config['CONTROLREGION']['verticesx']
    vertsy = config['CONTROLREGION']['verticesy']
    anncentx, anncenty = vertsx[0], vertsy[0]
    annrad = np.sqrt( (vertsx[0]-vertsx[2])**2+
                      (vertsy[0]-vertsy[2])**2)
    
    plt.ion()
    #This should be a function
    fig = plt.figure(figsize = (10, 10))
    ax1 =plt.subplot2grid((4,4),(0, 0), rowspan =2, colspan = 2)
    ax2 = plt.subplot2grid((4,4),(0, 2), rowspan =2, colspan = 2)
    ax3 =plt.subplot2grid((4,4),(2, 0), rowspan =3, colspan = 2)
    ax4 =plt.subplot2grid((4,4),(2, 2), rowspan =3, colspan = 2)
    
    title = fig.suptitle('Speckle destruction')
    ax1.set_title('Image')
    ax2.set_title('Control region')
    ax3.set_title('RMS in region')
    ax4.set_title('Raw 1s contrast: ref '+
                    str(config['NULLING']['referenceval']))

    w1 = ax1.imshow(np.log(np.abs(defaultim)), origin='lower', interpolation = 'nearest')
    ax1.set_xlim(anncentx-annrad, anncentx+annrad)
    ax1.set_ylim(anncenty-annrad, anncenty+annrad)
    w2 = ax2.imshow(np.log(np.abs(controlregion*defaultim)), origin='lower', interpolation = 'nearest')
    ax2.set_xlim(anncentx-annrad, anncentx+annrad)
    ax2.set_ylim(anncenty-annrad, anncenty+annrad)
    
    w3 = ax3.plot(np.arange(N_iterations),np.repeat(min_rms, N_iterations), 'k.')
    ax3.set_xlim(0, N_iterations)
    
    w4 = ax4.plot(min_contrastcurve[0], min_contrastcurve[1], 'k.')
     
    itcounter  = []
    maxfluxes = []
    meanfluxes = []
    totalfluxes = []
    rmsfluxes = []
    
    plt.show()
    tstamp = time.strftime("%Y%m%d-%H%M%S").replace(' ', '_')
    cubeoutputdir = os.path.join(null_params['outputdir'],
                                 tstamp)
    
    if not os.path.exists(cubeoutputdir):
        os.makedirs(cubeoutputdir)
    sys.stdout = Logger(os.path.join(cubeoutputdir, 'logfile.txt'))
    print("CUBE OUTPUT DIR", cubeoutputdir)
    print("making resultcubes 1")
    result_imagecube =  output_imagecube(
                           N_iterations, defaultim, 
                           filepath = os.path.join(cubeoutputdir,
                                        'test_'+ tstamp+'.fits'),
                           comment = ' ' , 
                           configfile = configfilename)
    
    print("making resultcubes 2")
    clean_imagecube=  output_imagecube(
                           N_iterations, defaultim, 
                           filepath = os.path.join(cubeoutputdir,
                                        'test_clean_'+ tstamp+'.fits'),
                           comment = ' ',
                           configfile = configfilename)
    
    print("making resultcubes 3")
    cal_imagecube = output_imagecube(
                            4, defaultim, 
                           filepath = os.path.join(cubeoutputdir,
                                         'test_cals_'+ tstamp+'.fits'),
                           comment = ' ',
                           configfile = configfilename)
    
    print("making flatmap imagecube")
    dmsize = int(config['AOSYS']['dmcyclesperap']*2)
    dmshape_cube = output_imagecube(
                           N_iterations, 
                           np.ones((dmsize, dmsize)), 
                           filepath = os.path.join(cubeoutputdir,
                                        'flatmap'+ tstamp+'.fits'),
                           comment = 'flatmaps', 
                           configfile = configfilename)
    cal_imagecube.update(controlregion)
    cal_imagecube.update(bgds['bkgd'])
    cal_imagecube.update(bgds['masterflat'])
    cal_imagecube.update(bgds['badpix'])
   
   
    ####MAIN LOOP STARTS HERE##### 
    print("BEGINNING NULLING LOOP" )
     
    recompute_center = config['QACITS']['recenter_qacits']
    #NEEDTOFIX MAKE SURE 15 and 20 ARE SAFE AND DON'T OVERRRUN IMAGE
    bgd_pix = pro.annulus(np.zeros(defaultim.shape),
                           config['IM_PARAMS']['centerx'],
                           config['IM_PARAMS']['centery'],
                           int(15*config['IM_PARAMS']['lambdaoverd']),
                           int(20*config['IM_PARAMS']['lambdaoverd']))
    wherebgd = np.where(bgd_pix)
    for iteration in range(N_iterations):
        itcounter.append(iteration)

        # current_dm_solution = AOsystem.get_dm_shape()
        current_dm_solution = KeckSim.get_dm_shape()
        print('poildecul')
        print(np.max(current_dm_solution))
        dmshape_cube.update(current_dm_solution)
                    
        print("Taking image of speckle field")
        # raw_im = Detector.take_image()
        raw_im = KeckSim.take_image()
        if update_bgd == True:
            updatelev = np.median(raw_im[wherebgd])/np.median(bgds['bkgd'][wherebgd])
            bgds['bkgd'] = bgds['bkgd']*updatelev
            print("scaling background by ", updatelev)

        print(raw_im.shape)
        result_imagecube.update(raw_im)
        cleaned_image = pre.equalize_image(raw_im, **bgds)
        clean_imagecube.update(cleaned_image)
        field_ctrl = cleaned_image*controlregion
        meanfluxes.append(np.mean(field_ctrl[field_ctrl>0]))

        maxfluxes.append(np.max(field_ctrl[field_ctrl>0]))
        totalfluxes.append(np.sum(field_ctrl))
        rmsfluxes.append(np.std(field_ctrl[field_ctrl>0])/
                         config['NULLING']['referenceval'])
        
        if config['NULLING']['scale_intensity']:
            innerim = cleaned_image*innerctrlann
            basemedianflux = np.median(innerim[innerim>0])
            print("Base median flux in inner region is ", basemedianflux)
        
        ax3.plot(itcounter,rmsfluxes) 
        
        
        pixrad, clevel = cc.contrastcurve_simple(cleaned_image,
                                   cx = config['IM_PARAMS']['centerx'],
                                   cy = config['IM_PARAMS']['centery'],
                                    region = controlregion,
                                    robust = True, fwhm = 6.0,
                                    maxrad = 50)
        ax4.plot(pixrad, clevel/float(
                        config['NULLING']['referenceval']))
        
        ax1.set_title('Iteration: '+str(iteration))

        border = np.abs(sciim.filters.laplace(controlregion))
        border[border>0] = np.nan
        w1.set_data(np.log(np.abs(field_ctrl)))
        w1.autoscale()
        w1.set_data(np.log(np.abs(cleaned_image))+border)
        #this works
        plt.draw()
        plt.pause(0.02) 
        # Check if there is an improvement
        if iteration >0:
            ##FIX THIS
            config = flh.validate_configfile(soft_ini, soft_spec)    
            printstats(cleaned_image, speckleslist) 
            flh.writeout(current_dm_solution, 'latestiteration.fits')
            
            if recompute_center == True:
                print("Recentering")
                #compute offset in lambda/d units
                lam_off_est = qac.qacits_estimate_tiptilt(cleaned_image,
                                        config['QACITS']['inner_rad_pix'], 
                                        config['QACITS']['outer_rad_pix'], 
                                        config['QACITS']['psfamp'], 
                                        cx = config['QACITS']['setpointx'],
                                        cy = config['QACITS']['setpointy'],
                                        small_tt_model=['outer','linear', 0.08],
                                        large_tt_model=['outer','linear', 0.08])
                if np.linalg.norm(lam_off_est)>1.5:
                    print(("Too much offset: ", lam_off_est, " not correcting"))
                    lam_off_est = np.array([0, 0])     
                pixeloffset = (-1*lam_off_est*
                               config['IM_PARAMS']['lambdaoverd']*
                               config['QACITS']['gain'])
                print("pixeloffset computed: ", pixeloffset)
                AOsystem.dtoff(pixeloffset)
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
           ' total_intensity: '+str(np.sum(field_ctrl)))
        #return a list of points
        print("computing interesting bright spots")
        
        #note indices and coordinates are reversed
        if config['DETECTION']['manual_click']:
            user_inp = input('Type coordinates of points to null as follows: 400, 600; 405, 610; etc:  ')
            ijofinterest = [x.split(',') for x in user_inp.split(';')]
            ijofinterest = [np.round(np.array(x, dtype = np.float)).astype(int) for x in ijofinterest]
            ijofinterest = [tuple(x) for x in ijofinterest]
            xyofinterest = ijofinterest
        else:
            ijofinterest = identify_bright_points(field_ctrl)
            xyofinterest = [p[::-1] for p in ijofinterest] 
        
        print("computed ",str(len(xyofinterest)), " bright spots")
        max_specks = config['DETECTION']['max_speckles']
        
        if len(xyofinterest)<max_specks:
            max_specks = len(xyofinterest)

        fps = filterpoints(xyofinterest, 
                           max = max_specks, 
                           rad=config['NULLING']['exclusionzone'],
                           cx = config['IM_PARAMS']['centerx'],
                           cy = config['IM_PARAMS']['centery'])
        print("creating speckle objects")
        speckleslist = [speckle(cleaned_image, xy[0], xy[1], config) for xy in fps]
        phases = null_params['phases']
        
        for idx, phase in enumerate(phases):
            print("Phase ", phase)
            phaseflat = 0
            allspeck_aps= 0
            #put in sine waves at speckle locations
            for speck in speckleslist:
                #XXX
                phaseflat= phaseflat+speck.generate_flatmap(phase)
                
                allspeck_aps = allspeck_aps+ speck.aperture
            
            ax2.set_title('Phase: '+str(phase))
            if idx == 0:
                w1.set_data( (allspeck_aps*cleaned_image)-0.95*cleaned_image + border)
                w1.autoscale()
                plt.draw()
            
            # status = AOsystem.set_dm_shape(current_dm_solution +phaseflat)

            status = KeckSim.set_dm_shape(current_dm_solution +phaseflat)
            
            # phaseim = Detector.take_image()
            phaseim = KeckSim.take_image()
            if update_bgd == True:
                updatelev = np.median(raw_im[wherebgd])/np.median(bgds['bkgd'][wherebgd])
                bgds['bkgd'] = bgds['bkgd']*updatelev
                print("scaling background by ", updatelev)
            phaseim = pre.equalize_image(phaseim, **bgds) 
            w1.set_data(phaseim+border)
            w1.autoscale()
            plt.draw()
            if recompute_center == True:
                print("Recentering")
                #compute offset in lambda/d units
                lam_off_est = qac.qacits_estimate_tiptilt(phaseim,
                                        config['QACITS']['inner_rad_pix'], 
                                        config['QACITS']['outer_rad_pix'], 
                                        config['QACITS']['psfamp'], 
                                        cx = config['QACITS']['setpointx'],
                                        cy = config['QACITS']['setpointy'],
                                        small_tt_model=['outer','linear', 0.08],
                                        large_tt_model=['outer','linear', 0.08])
                if np.linalg.norm(lam_off_est)>1.5:
                    print(("Too much offset: ", lam_off_est, " not correcting"))
                    lam_off_est = np.array([0, 0])     
                pixeloffset = (-1*lam_off_est*
                               config['IM_PARAMS']['lambdaoverd']*
                               config['QACITS']['gain'])
                print("pixeloffset computed: ", pixeloffset)
                # AOsystem.dtoff(pixeloffset)
            
            if config['NULLING']['scale_intensity']:
                phaseimmedianflux = np.median((phaseim*innerctrlann)[innerctrlann>0])
                print("median flux in region is ", phaseimmedianflux)
                factor = basemedianflux/phaseimmedianflux
                print("scaling image by ", factor)
                phaseim = phaseim*factor

            w2.set_data(np.log(np.abs(phaseim*controlregion)))
            w2.autoscale();plt.draw();plt.pause(0.02) 
            
            print("recomputing intensities")
            for speck in speckleslist:
                phase_int = speck.recompute_intensity(phaseim)
                speck.phase_intensities[idx] = phase_int

        # AOsystem.set_dm_shape(current_dm_solution)
        KeckSim.set_dm_shape(current_dm_solution)

        if config['NULLING']['null_gain'] == False:
            defaultgain = config['NULLING']['default_flatmap_gain']
            nullmap= generate_phase_nullmap(speckleslist, defaultgain) 
            # AOsystem.set_dm_shape(current_dm_solution + nullmap)
            KeckSim.set_dm_shape(current_dm_solution + nullmap)

        
        if config['NULLING']['null_gain'] == True:       
            ##NOW CALCULATE GAIN NULLS 
            print("DETERMINING NULL GAINS")
            gains = config['NULLING']['amplitudegains']
            for idx, gain in enumerate(gains):
                print("Checking optimal gains")
                nullmap= generate_phase_nullmap(speckleslist,  gain) 
                # AOsystem.set_dm_shape(current_dm_solution + nullmap)
                KeckSim.set_dm_shape(current_dm_solution + nullmap)
                ampim = KeckSim.take_image()
                
                if update_bgd == True:
                    updatelev = np.median(raw_im[wherebgd])/np.median(bgds['bkgd'][wherebgd])
                    bgds['bkgd'] = bgds['bkgd']*updatelev
                    print("scaling background by ", updatelev)
                ampim = pre.equalize_image(ampim, **bgds) 
                
                if recompute_center == True:
                    print("Recentering")
                    #compute offset in lambda/d units
                    lam_off_est = qac.qacits_estimate_tiptilt(ampim,
                                            config['QACITS']['inner_rad_pix'], 
                                            config['QACITS']['outer_rad_pix'], 
                                            config['QACITS']['psfamp'], 
                                            cx = config['QACITS']['setpointx'],
                                            cy = config['QACITS']['setpointy'],
                                            small_tt_model=['outer','linear', 0.08],
                                            large_tt_model=['outer','linear', 0.08])
                    if np.linalg.norm(lam_off_est)>1.5:
                        print(("Too much offset: ", lam_off_est, " not correcting"))
                        lam_off_est = np.array([0, 0])     
                    pixeloffset = (-1*lam_off_est*
                                   config['IM_PARAMS']['lambdaoverd']*
                                   config['QACITS']['gain'])
                    print("pixeloffset computed: ", pixeloffset)
                    # AOsystem.dtoff(pixeloffset)
                
                w2.set_data(np.log(np.abs(ampim*controlregion)))
                ax2.set_title('Gain: '+str(gain))
                w2.autoscale();plt.draw();plt.pause(0.02) 
                for speck in speckleslist:
                    amp_int = speck.recompute_intensity(ampim)
                    speck.gain_intensities[idx] = amp_int
            for speck in speckleslist:
                speck.compute_null_gain()
            supernullmap = generate_super_nullmap(speckleslist)
            print("Loading supernullmap now that optimal gains have been found!")
            # AOsystem.set_dm_shape(current_dm_solution + supernullmap)
            KeckSim.set_dm_shape(current_dm_solution + supernullmap)
            plt.draw()

    print("Max iterations reached.  Exiting gracefully.  Run again if you like")
