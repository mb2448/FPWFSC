from common import support_functions as sf
import sn_functions as snf
import ipdb

class SpeckleAreaNulling:
    def __init__(self, camera=None, aosystem=None, initial_probe_amplitude=None, initial_regularization=None,
                 controlregion_iwa=None, controlregion_owa=None, 
                 xcenter=None, ycenter=None, Npix_foc=None, lambdaoverD=None,
                 flipx=None, flipy=None, rotation_angle_deg=None):
        
        self.camera = camera
        self.aosystem = aosystem
        self.initial_probe_amplitude = initial_probe_amplitude
        self.regularization = initial_regularization
        self.probe_amplitude = initial_probe_amplitude
        
        # NOTE: Same dimensions as image
        # NOTE: Referenced to same origin as image

        self.xcenter = xcenter
        self.ycenter = ycenter
        self.lambdaoverD = lambdaoverD
        
        self.controlregion_iwa = controlregion_iwa
        self.controlregion_owa = controlregion_owa
        self.controlregion_iwa_pix = self.controlregion_iwa* \
                                     self.lambdaoverD
        self.controlregion_owa_pix = self.controlregion_owa*\
                                    self.lambdaoverD
        self.imparams = {
                'npix': Npix_foc,
                'xcen': self.xcenter,
                'ycen': self.ycenter,
                'flipx': flipx,
                'flipy': flipy,
                'rotation_angle_deg': rotation_angle_deg,
        }

        # Take reference image
        self.rawI0 = self.camera.take_image()
        self.controlregion = snf.create_annular_wedge(self.rawI0, 
                                              self.imparams['xcen'], 
                                              self.imparams['ycen'], 
                                              self.controlregion_iwa_pix, 
                                              self.controlregion_owa_pix, 
                                              -90, 90)
        #create_annular_wedge(image, xcen, ycen, rad1, rad2, theta1, theta2)
        self.I0 = sf.reduce_images(self.rawI0, **self.imparams)
        # Construct the probes

        
    def _measure(self):
        pass

    def iterate(self, probe_amplitude=None, regularization=None):
        
        if probe_amplitude is None:
            self.probe_amplitude = probe_amplitude
        
        if regularization is None:
            self.regularization = regularization
        pass
    