from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import sn_math as snm
import numpy as np
import dm_functions as DM
# import sn_hardware as snh
import sn_sims as sns
import sn_filehandling as flh
import sn_preprocessing as pre
import scipy.optimize as opt
from detect_speckles import create_speckle_aperture, get_speckle_photometry
import time
import pdb

def onpress( event):
    xi, yi = (int(round(n)) for n in (event.xdata, event.ydata))
    print (xi, yi)
    xypixels.append( (xi, yi))
    if len(xypixels) == 4:
        fig.canvas.mpl_disconnect(cid)
    pass
    #return (xi, yi )

if __name__ == "__main__":
    # Initialized config and spec file names
    soft_ini  = 'Config/SN_Software.ini'
    soft_spec = 'Config/SN_Software.spec'
    hard_ini  = 'Config/SN_Hardware.ini'
    hard_spec = 'Config/SN_Hardware.spec'

    # Read the config files and check if satisfied spec files requierements.
    config = flh.validate_configfile(soft_ini, soft_spec)

    # Hardware Connection
    print('')
    print('############################################################')
    print('#################### Hardware Connection ###################')
    print('############################################################')
    print('')

    # Instancie AO system and Detector selected
    #AOsystem = snh.KECK2AO('KECKAO', hard_ini, hard_spec)
    #Detector = snh.NIRC2(  'NIRC2' , hard_ini, hard_spec)
    KeckSim = sns.FakeCoronagraph()
    KeckSim.make_DM()
    KeckSim.make_TTM()
    x0 = np.array([0.0,0.0])
    KeckSim.set_TTM(x0)
    # KeckSim.make_aberration(0.001)
    KeckSim.make_aberration(0.1)

    # Beginning of the setup verification
    print('')
    print('############################################################')
    print('################ Beginning DM INTENSITY CALIBRATION #################')
    print('############################################################')
    print('')

    # Read the detector calibration data.
    # To acquire detector valibration data use the script:
    # Detector_Calibration.py
    print('Read the calibration data acquired previously.')
    bgds = flh.setup_bgd_dict(config)

    intconf = config['INTENSITY_CAL']
    im_params = config['IM_PARAMS']
    centerx = config['IM_PARAMS']['centerx']
    centery = config['IM_PARAMS']['centery']

    DMamp = float(intconf['default_dm_amplitude'])
    ap_rad = intconf['aperture_radius']

    initial_dm_shape = KeckSim.get_dm_shape()
    firstim = KeckSim.take_image()#works, tested
    firstim = pre.equalize_image(firstim, **bgds)

    #initialize range to test
    kr = np.arange(intconf['min'],
                   intconf['max'],
                   intconf['stepsize'])
    print(kr)
    #Set up display area
    plt.ion()
    fig = plt.figure(figsize = (12,12))
    ax1 =plt.subplot2grid((4,4),(0, 0), rowspan =3, colspan = 3)
    ax2 =plt.subplot2grid((4,4),(3, 0))
    ax3 =plt.subplot2grid((4,4),(3, 3))
    title = fig.suptitle('Intensity Calibration')
    ax1.set_title('Image-Original Image')
    ax2.set_title('Default DM map')
    ax3.set_title('Map perturbation')
    w1 = ax1.imshow(np.log(np.abs(firstim)), origin='lower', interpolation = 'nearest')
    #ax1.set_xlim((centerx - 256, centerx+256))
    #ax1.set_ylim((centery - 256, centery+256))
    w2 = ax2.imshow(initial_dm_shape, origin='lower', interpolation = 'nearest')
    w3 = ax3.imshow(initial_dm_shape, origin='lower', interpolation = 'nearest')
    plt.show()

    xypixels = []
    ximcoords, yimcoords = np.meshgrid(np.arange(firstim.shape[0]),
                                      np.arange(firstim.shape[1]))

    actual_kvector_array = []
    intensity_array = []

    for kvecr in kr:
        if not intconf['auto']:
            title.set_text("Click on the 4 spots, then hit enter.  Kr: "+str(kvecr))
        if intconf['auto']:
            title.set_text("Kr: "+str(kvecr))
        #Check this line
        phase = np.pi/2
        additionmapx = DM.make_speckle_kxy(kvecr, 0,DMamp , phase)
        additionmapy = DM.make_speckle_kxy(0,kvecr, DMamp, phase)
        additionmap = additionmapx + additionmapy

        status = KeckSim.set_dm_shape((initial_dm_shape + additionmap))
        im = KeckSim.take_image()
        im = pre.equalize_image(im, **bgds)
        #removes speckle field to first order
        im = im - firstim

        w3.set_data(additionmap)
        # w1.set_data(np.log(np.abs(im)))
        w1.set_data(np.abs(im))

        plt.draw()
        plt.pause(0.01)
        if not intconf['auto']:
            print("This probably does not work")
            cid = fig.canvas.mpl_connect('button_press_event', onpress)
            w = raw_input("Click the four spots then press enter in this window\n")
        if intconf['auto']:
            xy0 = DM.convert_kvecs_pixels(kvecr, 0, **im_params)
            xy1 = DM.convert_kvecs_pixels(-kvecr, 0, **im_params)
            xy2 = DM.convert_kvecs_pixels(0,kvecr, **im_params)
            xy3 = DM.convert_kvecs_pixels(0,-kvecr, **im_params)
            xypixels = [xy0, xy1, xy2, xy3]

        print("Fitting spot amplitudes and positions")
        #loops over the spots you clicked

        meankvec = 0
        meanphotom = 0
        for idx, xy in enumerate(xypixels):
            subim = pre.subimage(im, xy, window = 10)
            subx  = pre.subimage(ximcoords, xy, window = 10)
            suby  = pre.subimage(yimcoords, xy, window = 10)
            gauss_params = snm.fitgaussian(subim, subx, suby)
            print("Gauss_params",gauss_params)
            spotx, spoty = gauss_params[1], gauss_params[2]
            print("Spotx, spoty", spotx, spoty)
            #convert pixels returns an xy pair.
            # in theory all the kvecs should be (x, 0) or (0, y)
            # in which case the norm will be equal to x or y
            kvec = np.linalg.norm(
                    DM.convert_pixels_kvecs(spotx,
                                           spoty,
                                           **im_params))
            print( "Gaussian centroid", spotx, spoty)
            print( "corr. k-vector", kvec)
            meankvec = (meankvec*float(idx)/float(idx+1) +
                             kvec/float(idx+1))
            aperture = create_speckle_aperture(im,spotx, spoty,ap_rad)
            photometry = get_speckle_photometry(im, aperture)

            w1.set_data(np.log(np.abs(im*aperture)))
            plt.draw()
            plt.pause(0.01)
            w1.set_data(np.log(np.abs(im)))
            plt.draw()
            plt.pause(0.01)
            #w1.autoscale_view()
            meanphotom = (meanphotom*float(idx)/float(idx+1) +
                                 photometry/float(idx+1))

        print( "\nIntended radial k-vector: "+str(kvecr) +
               "\nMean k-vector: " + str(meankvec))

        intensity_array.append(meanphotom)
        actual_kvector_array.append(meankvec)
        xypixels = []

    plt.ioff()
    plt.close('all')
    print ("\n Intended K-vectors")
    print (kr)
    print ("\n Actual K-vectors")
    print (actual_kvector_array)
    print ("\n Intensities")
    print (intensity_array)

    #FIT THE FUNCTION
    print( "\n Performing global fit")
    popt, pcov = opt.curve_fit(lambda x, ra, rb, rc: DM.intensitymodel(DMamp, x, a=ra, b=rb, c=rc),
                                np.array(actual_kvector_array), np.array(intensity_array))
    fita, fitb, fitc = popt[0], popt[1], popt[2]
    print(fita, fitb, fitc)

    print( "Writing out a,b,c fit parameters to configfile")
    config['INTENSITY_CAL']['abc']['a'] = fita
    config['INTENSITY_CAL']['abc']['b'] = fitb
    config['INTENSITY_CAL']['abc']['c'] = fitc
    config.write()

    print("Reloading initial flatmap")
    KeckSim.set_dm_shape(initial_dm_shape)
    smoothks = np.arange(min(actual_kvector_array), max(actual_kvector_array), .01)

    plt.plot(actual_kvector_array, intensity_array, '.')
    plt.plot(smoothks, DM.intensitymodel(DMamp, smoothks, a=fita, b=fitb, c=fitc))
    plt.xlabel('actual k-vectors')
    plt.ylabel('intensity [counts]')
    plt.title('Data and fit')
    plt.show()

    plt.plot(actual_kvector_array, [x/DMamp**2 for x in intensity_array], '.')
    plt.plot(smoothks, 1.0/DMamp**2*DM.intensitymodel(DMamp, smoothks, a=fita, b=fitb, c=fitc))
    plt.xlabel('actual k-vectors')
    plt.ylabel('Model fit/amp**2; this shoudl be the same for every cal')
    plt.title('DM amp: '+str(DMamp))
    plt.show()

    #qacits_val = DM.intensitymodel(0.3, 5, fita, fitb, fitc)
    #qacits_mult= 900.0
    ####Note: aperture photometry said the psf was 40e6 counts when
    ##the intensity of a spot at 5l/D with 300 nm was 125,000 cts
    ##however, the psf amp that comes out of that is below what i was expecting
    ##by a factor of 30 or so.  Either the linear estimator coefficient is off,
    ##or something else i don't understand is jacked up.
    #print "QACITS default intensity setpoint at 300 nm, 5l/D: ",qacits_val
    #print "SETTING QACITS PSF AMP TO: ", qacits_val*qacits_mult
    #config['QACITS']['psfamp'] = qacits_val*qacits_mult
    #config.write()
