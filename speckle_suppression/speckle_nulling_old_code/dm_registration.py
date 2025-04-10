############################### Import Library ################################

## Math Library
import numpy as np
## System library
import sys
## Operating system library
import os
## .fits file library
import astropy.io.fits as fits
## Library used to plot graphics and show images
import matplotlib.pyplot as plt

# Location of the FIU library
#sys.path.append('/kroot/src/kss/nirspec/nsfiu/dev/lib/')
## Function use to control NIRC2
#import Nirc2_cmds as Nirc2
## Function use to get the path where data should be saved
#from FIU_Commands import get_path

# Location of the Speckle Nulling library
# sys.path.append('/kroot/src/kss/nirspec/nsfiu/dev/speckle_nulling/')
sys.path.append('/pueyo/CodingProjects/speckle_nulling_LAP/')
## Libraries for simulator speckle nulling data
import sn_sims as sns
## Libraries to process speckle nulling data
import sn_preprocessing as pre
## Libraries to handle speckle nulling files
import sn_filehandling  as flh
## Libraries to import hardware used by speckle nulling
# import sn_hardware as snh
## Libraries to create dm shapes
import dm_functions as dm
## Libraries use to perform some math operation specific to speckle nulling
import sn_math as snm
from detect_speckles import create_speckle_aperture, get_speckle_photometry, \
get_total_aperture_flux


from configobj import ConfigObj
import sn_filehandling as flh
#import flatmapfunctions as FM
from validate import Validator
#import flatmapfunctions_keck as fmf
import dm_functions as DM
import time
import scipy.ndimage as sciim
import scipy.optimize as opt
import astropy.io.fits as pf

############################## Local Definitions ##############################

# =============================================================================
def recenter_sld(image, spots, window = 20):
    ''' -----------------------------------------------------------------------
    ----------------------------------------------------------------------- '''
    #slightly less dumb recentering
    scs = np.zeros((len(spots), 2))
    for idx, xy in enumerate(spots):
        subim = pre.subimage(image, xy, window=window)
        x, y = np.meshgrid( np.arange(subim.shape[0]),
                            np.arange(subim.shape[1]))
        dilation = sciim.filters.maximum_filter(subim, 5)
        maxcluster = np.where(dilation == np.max(dilation))
        xdata = x[maxcluster]
        ydata = y[maxcluster]
        ptstofit = subim[maxcluster]
        initguess = (np.max(ptstofit)-np.min(ptstofit),
                     xdata[np.where(ptstofit == np.max(ptstofit))],
                     ydata[np.where(ptstofit == np.max(ptstofit))],
                     1.0,
                     1.0,
                     0.0,
                     np.min(ptstofit))
        try:
            popt, pcov = opt.curve_fit(snm.twoD_Gaussian, (xdata,ydata),
                                        ptstofit.ravel(),
                                        p0 = initguess, maxfev = 100000000)
        except:
            ipdb.set_trace()
        xcenp = popt[1]
        ycenp = popt[2]
        xcen = xy[0]-round(window/2)+xcenp
        ycen = xy[1]-round(window/2)+ycenp
        scs[idx,:] = xcen, ycen
    return scs

# =============================================================================
#def recenter_satellites(image, spots, window=20):
#    ''' -----------------------------------------------------------------------
#    Centroid each satellite spot using a 2d gaussian
#    ----------------------------------------------------------------------- '''
#
#    #satellite centers
#    scs = np.zeros((len(spots), 2))
#    for idx,xy in enumerate(spots):
#        subim = pre.subimage(image, xy, window=window)
#        popt = snm.image_centroid_gaussian(subim)
#        xcenp = popt[1]
#        ycenp = popt[2]
#        if np.abs(xcenp-window//2)>window//2:
#            print( "center drifted too much, using previous")
#            xcenp = window//2
#        if np.abs(ycenp-window//2)>window//2:
#            ipdb.set_trace()
#            print( "center drifted too much, using previous")
#            ycenp = window//2
#        xcen = xy[0]-round(window/2)+xcenp
#        ycen = xy[1]-round(window/2)+ycenp
#        scs[idx,:] = xcen, ycen
#    return scs

# =============================================================================
def get_satellite_centroids(image, window = 20, cmt=None, guess_spots=None):
    ''' -----------------------------------------------------------------------
    Compute the centroid of each satellite spot using a 2d gaussian fit.
    ----------------------------------------------------------------------- '''
    # Prepare comments for the get_spot_locations function
    if cmt is None:
        cmt  = 'SHIFT-Click on the satellites CLOCKWISE'
        cmt += '\n Starting from 9 o clock.'
        cmt += '\n When done close the window.'

    if guess_spots is None:
        print('')
        print('Positions found by user.')
        # Get an estimation of the position of each spot
        guess_spots = pre.get_spot_locations(image, eq=True, comment= cmt)

    scs = np.zeros((len(guess_spots), 2))
    # Measure acurately the position of each satellite
    print('')
    print('Positions adjusted by gaussian fit.')
    # Compute the position of each satelite by fitting a 2d gaussian on the
    # Speckle previously identified by the user,
    for idx,xy in enumerate(guess_spots):
        # idx contains the number of the speckle
        # xy the position of the speckle.

        # Generate a sub image centered on the xy position
        y0    = int(round(xy[0]))
        x0    = int(round(xy[1]))
        hw    = int(round(window/2.))
        subim =  image[x0-hw:x0+hw,y0-hw:y0+hw]
        # Fit a gaussian on the subimage roughtly centered on the speckle.
        popt  = snm.image_centroid_gaussian(subim)
        # Extract center position (x,y) of the gaussian fitted on the speckles.
        xcen  = round(y0-hw+popt[0],3)
        ycen  = round(x0-hw+popt[1],3)
        # Add the position computed to the list of positions.
        scs[idx,:] = xcen, ycen
    # Return the list of positions
    return scs

# =============================================================================
def get_center_from_singlespot(spotcenter, spotkvec, lambdaoverd = None):
    ''' -----------------------------------------------------------------------
    ----------------------------------------------------------------------- '''
    kx, ky = spotkvec
    sx, sy = spotcenter
    cx = sx - kx*lambdaoverd
    cy = sy - ky*lambdaoverd
    #print( cx, cy)
    return (cx, cy)

# =============================================================================
def find_center_all(spotcenters, kvecs, lambdaoverd = None,
                    max_drift=None, curr_cent = None):
    ''' -----------------------------------------------------------------------
    Gets estimate of the center position from each spot, discards outliers,
    and returns an estimate
    ----------------------------------------------------------------------- '''
    center_ests = []
    for idx, spotcent in enumerate(spotcenters):
        c = get_center_from_singlespot(spotcent,
                                       kvecs[idx],
                                       lambdaoverd=lambdaoverd)
        print("center from singlespot", c)
        if max_drift is not None:
            if np.linalg.norm(np.array(curr_cent) - np.array(c))>max_drift:
                print( "discarding ",c," as too far from current")
                continue
        else:
            center_ests.append(c)
    return np.median(np.array(center_ests), axis = 0)

# =============================================================================
def find_center(centers):
    ''' -----------------------------------------------------------------------
    returns the mean of the centers WTF we need a function to perform this task
    ----------------------------------------------------------------------- '''
    return np.mean(centers, axis = 0)

# =============================================================================
def find_center_2(centers):
    ''' -----------------------------------------------------------------------
    ----------------------------------------------------------------------- '''
    xc = (centers[0][0] + centers[2][0])/2
    yc = (centers[1][1] + centers[3][1])/2
    return np.array([xc, yc])

# =============================================================================
def find_angle(centers):
    ''' -----------------------------------------------------------------------
    Uses the four centers, presumably square, to find the rotation angle of the
    DM
    ----------------------------------------------------------------------- '''
    center = np.mean(centers, axis = 0)
    reltocenter = centers-center
    angs = [np.arctan(reltocenter[i,1]/reltocenter[i,0])*180/np.pi for i in range(4)]
    for idx,ang in enumerate(angs):
        if np.abs(ang-90)<3:
            angs[idx]= ang-90.0
    print( "WARNING: SOME TRICKERY EXISTS HERE")
    print( "MIKE OVERRIDES W/90 DEGREES LOL")
    #angs = np.array(angs)+np.array([90.0, 90, 90, 90])
    #assert np.std(angs)< 3.0
    #angle = np.mean(angs)
    return 90.0

# =============================================================================
def get_lambdaoverd(centroids, cyclesperap):
    ''' -----------------------------------------------------------------------
    Calcualtes lambda/d by taking the average of the two diagonas of the
    square, then divides by cycles per aperture, then divides by 2
    ----------------------------------------------------------------------- '''
    diag1dist = np.linalg.norm(centroids[0,:] -centroids[2,:])
    diag2dist = np.linalg.norm(centroids[1,:] -centroids[3,:])
    avgdiagdist = 0.5*(diag1dist+diag2dist)
    return avgdiagdist/2/cyclesperap

def get_satellite_center_and_intensity(cleanimage,
                                       configfilename,
                                       configspecfile):
    """Uses the four satellite speckles to figure out the center of the
    image and the mean flux of the satellite speckles.  Uses the configfile to
    figure out a lot of these parameters, but this can be changed into
    explicit function inputs in the future if desired"""
    config = ConfigObj(configfilename, configspec=configspecfile)
    val = Validator()
    check = config.validate(val)
    try:
        spot_guesses = [ config['CALSPOTS']['spot10oclock'],
                         config['CALSPOTS']['spot1oclock'],
                         config['CALSPOTS']['spot4oclock'],
                         config['CALSPOTS']['spot7oclock'],]
    except:
        print( "WARNING: SPOTS NOT FOUND IN CONFIGFILE. RECALCULATING")
        spot_guesses = pre.get_spot_locations(cleanimage, eq=True,
                comment='SHIFT-Click on the satellites CLOCKWISE'+
                         'starting from 9 o clock,\n then close the window')

    lod = config['IM_PARAMS']['lambdaoverd']
    kvec = config['CALSPOTS']['wafflekvec']
    curr_cent = (config['IM_PARAMS']['centerx'],
                 config['IM_PARAMS']['centery'])
    ap_rad = config['INTENSITY_CAL']['aperture_radius']

    if config['CALSPOTS']['mode']=='waffle':
        spot_kvecs = [(-1*kvec, 1*kvec),
                      (1*kvec, 1*kvec),
                      (kvec, -1*kvec),
                      (-1*kvec, -1*kvec)]

    if config['CALSPOTS']['mode']=='square':
        spot_kvecs = [(-1*kvec, 0),
                      (0, kvec),
                      (kvec, 0),
                      (0, -1*kvec)]


    spotcenters = get_satellite_centroids(cleanimage, guess_spots=spot_guesses)
    print("Spotcenters:, ", spotcenters)
    spotintensities = []
    for spotcenter in spotcenters:
        spotx, spoty = spotcenter
        aperture = create_speckle_aperture(cleanimage, spotx, spoty, ap_rad)
        mean_phot = get_speckle_photometry(cleanimage, aperture)
        spotintensities.append(mean_phot)
    #find the average, or best image center using the spots
    #note this can be done in multiple ways. your mileage may vary
    c = find_center_all(spotcenters, spot_kvecs,
                        lambdaoverd = lod,
                        curr_cent = curr_cent)
    intensity = np.mean(spotintensities)
    return c, intensity
# =============================================================================
def dm_reg_autorun(cleanimage, configfilename, configspecfile):
    ''' -----------------------------------------------------------------------
    ----------------------------------------------------------------------- '''
    #configfilename = 'speckle_null_config.ini'
    config = ConfigObj(configfilename, configspec=configspecfile)
    val = Validator()
    check = config.validate(val)
    try:
        spot_guesses = [ config['CALSPOTS']['spot10oclock'],
                         config['CALSPOTS']['spot1oclock'],
                         config['CALSPOTS']['spot4oclock'],
                         config['CALSPOTS']['spot7oclock'],]
    except:
        print( "WARNING: SPOTS NOT FOUND IN CONFIGFILE. RECALCULATING")
        spot_guesses = pre.get_spot_locations(cleanimage, eq=True,
                comment='SHIFT-Click on the satellites CLOCKWISE'+
                         'starting from 9 o clock,\n then close the window')

    lod = config['IM_PARAMS']['lambdaoverd']
    kvec = config['CALSPOTS']['wafflekvec']
    curr_cent = (config['IM_PARAMS']['centerx'],
                 config['IM_PARAMS']['centery'])
    ap_rad = config['INTENSITY_CAL']['aperture_radius']
    spot_kvecs = [(-1*kvec, 0),
                  (0, kvec),
                  (kvec, 0),
                  (0, -1*kvec)]
    spotcenters = get_satellite_centroids(cleanimage, guess_spots=spot_guesses)
    for spotcenter in spotcenters:
        spotx, spoty = spotcenter
        aperture = create_speckle_aperture(cleanimage, spotx, spoty, ap_rad)
        mean_phot = get_speckle_photometry(cleanimage, aperture)
        print("Spotcenter: ", spotcenter, ", Mean phot: ", mean_phot)
    #find the average, or best image center using the spots
    #note this can be done in multiple ways. your mileage may vary
    c = find_center_all(spotcenters, spot_kvecs,
                        lambdaoverd = lod,
                        curr_cent = curr_cent)
    #a =find_angle(spotcenters)

    #config['IM_PARAMS']['centerx'] = c[0]
    #config['IM_PARAMS']['centery'] = c[1]
    #config['IM_PARAMS']['angle']  = a
    #kvecr = config['CALSPOTS']['wafflekvec']

    #config['CALSPOTS']['spot10oclock'] = [np.round(x) for x in spotcenters[0]]
    #config['CALSPOTS']['spot1oclock'] = [np.round(x) for x in spotcenters[1]]
    #config['CALSPOTS']['spot4oclock'] = [np.round(x) for x in spotcenters[2]]
    #config['CALSPOTS']['spot7oclock'] = [np.round(x) for x in spotcenters[3]]
    print( "Image center: " , c)
    #print "lambda/D: ", str(lambdaoverd)
    #config.write()
    #print( "Updating configfile")
    return c
# =============================================================================


def generate_calspots_pattern(kvec=None, DMamp=None, mode=None):
    if mode == "square":
        # Prepare the DM offset for each set of speckles.
        speckle_x = DM.make_speckle_kxy(kvec,     0, DMamp, 0)
        speckle_y = DM.make_speckle_kxy(    0, kvec, DMamp, 0)

    if mode == "waffle":
        # Prepare the DM offset for each set of speckles.
        speckle_x = DM.make_speckle_kxy(kvec,  kvec, DMamp, 0)
        speckle_y = 0
    offset  = speckle_x + speckle_y
    return offset

if __name__ == "__main__":
    # sys.exit(1)
    ''' -----------------------------------------------------------------------
    This program performs DM registration.
    It takes an image with the DM flat, then an image with speckles as defined
    in the configuration file provided by the user. Then reload initial DM map.
    It subtracts the two, asks you to click on the satellites and then figures
    out lambda/d, the center and rotation of the image.
    It then saves these values to the configuration file provided by the user.
    ----------------------------------------------------------------------- '''

    # Initialized config and spec file names
    soft_ini  = 'Config/SN_Software.ini'
    soft_spec = 'Config/SN_Software.spec'
    hard_ini  = 'Config/SN_Hardware.ini'
    hard_spec = 'Config/SN_Hardware.spec'

    # Read the config files and check if satisfied spec files requierements.
    soft_config = flh.validate_configfile(soft_ini, soft_spec)

    # Hardware Connection
    print('')
    print('############################################################')
    print('#################### Hardware Connection ###################')
    print('############################################################')
    print('')

    KeckSim = sns.FakeCoronagraph()
    KeckSim.make_DM()
    x0 = np.array([0.0,0.0])
    KeckSim.make_TTM()
    KeckSim.set_TTM(x0)
    # KeckSim.make_aberration(0.001)
    KeckSim.make_aberration(0.1)
    # Instancie AO system and Detector selected
    # AOsystem = snh.KECK2AO('KECKAO', hard_ini, hard_spec)
    # Detector = snh.NIRC2(  'NIRC2' , hard_ini, hard_spec)


    # Beginning of the setup verification
    print('')
    print('############################################################')
    print('################ Beginning DM REGISTRATION #################')
    print('############################################################')
    print('')

    # Read the detector calibration data.
    # To acquire detector valibration data use the script:
    # Detector_Calibration.py
    print('Reading the calibration data acquired previously.')
    bgds = flh.setup_bgd_dict(soft_config)


    print('Get the inital shape of the DM.')
    # Get the DM shape currently apply to the DM
    # initial_dm_shape = AOsystem.get_dm_shape()
    initial_dm_shape = KeckSim.get_dm_shape()


    print('Take a first image without speckles.')
    # Acquire an image with the selected detector.
    # im_0 = Detector.take_image()
    im_0 = KeckSim.take_image()


    print('Prepare the DM map to poke two sets of speckles')
    # Get from the config file the amplitude and kvector of the speckles.

    DMamp = soft_config['CALSPOTS']['waffleamp' ]
    kvec = soft_config['CALSPOTS']['wafflekvec']
    mode = soft_config['CALSPOTS']['mode']
    print("Using mode ", mode)
    offset = generate_calspots_pattern(kvec=kvec, DMamp=DMamp, mode=mode)
    if mode == "square":
        kvecr = kvec
    if mode == "waffle":
        kvecr = kvec*np.sqrt(2)

    print ('Apply the map to the DM')
    # Apply the offset to the DM
    # AOsystem.set_dm_shape(initial_dm_shape + offset)
    KeckSim.set_dm_shape(initial_dm_shape + offset)


    print('Take a second image with speckles.')
    # Acquire an image with the selected detector.
    # im_1 = Detector.take_image()
    im_1 = KeckSim.take_image()

    #print ('NOT returning to the initial DM shape.')
    #print ('Leaving calibration spots on')
    #AOsystem.set_dm_shape(initial_dm_shape)

    print ('Return to the initial DM shape.')
    # AOsystem.set_dm_shape(initial_dm_shape)
    KeckSim.set_dm_shape(initial_dm_shape)


    # Compute the difference of the image acquired.
    image = im_1-im_0

    #
    spotcenters = get_satellite_centroids(image)
    print('caca')
    print(spotcenters)
    c =find_center(spotcenters)
    a =find_angle(spotcenters)

    soft_config['IM_PARAMS']['centerx'] = c[0]
    soft_config['IM_PARAMS']['centery'] = c[1]
    soft_config['QACITS']['setpointx'] = c[0]
    soft_config['QACITS']['setpointy'] = c[1]
    print( "RESETTING QACITS SETPOINT TO IMAGE CENTER!!!")
    soft_config['IM_PARAMS']['angle']  = a
    soft_config['CALSPOTS']['spot10oclock'] = [np.round(x) for x in spotcenters[0]]
    soft_config['CALSPOTS']['spot1oclock'] = [np.round(x) for x in spotcenters[1]]
    soft_config['CALSPOTS']['spot4oclock'] = [np.round(x) for x in spotcenters[2]]
    soft_config['CALSPOTS']['spot7oclock'] = [np.round(x) for x in spotcenters[3]]
    #cyclesperap = int(config['AOSYS']['dmcyclesperap'])
    lambdaoverd = get_lambdaoverd(spotcenters, kvecr)
    soft_config['IM_PARAMS']['lambdaoverd'] = lambdaoverd
    plt.imshow(image, origin = 'lower')
    plt.scatter(spotcenters[:,0], spotcenters[:,1], marker='x',color='r', alpha = 0.5)
    plt.scatter(c[0], c[1], marker='x',color='r')
    plt.title('Measured centers')
    plt.show()
    print( "Image center: " , c)
    print( "DM angle: ", a)
    print( "lambda/D: ", str(lambdaoverd))
    soft_config.write()

