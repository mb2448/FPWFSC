import sn_preprocessing as pre
import os
import matplotlib.pyplot as plt
import astropy.io.fits as pf
import numpy as np
from configobj import ConfigObj
import sn_filehandling as flh
import sn_processing as pro
import astropy.io.fits as fits


def define_control_annulus(image, cx= None, cy = None,
                           rad_in = None, rad_out = None):
    #"""SHIFT- Click on the image to define the vertices of a polygon defining a region. May be convex or concave"""
    #spots = pre.get_spot_locations(image,
    #        comment='SHIFT-click to select IN THIS ORDER, inner radius, and outer radius of the annular region to control')
    spots = [(cx +rad_in, cy), (cx+rad_out, cy)]
    xs, ys = np.meshgrid( np.arange(image.shape[1]),
                            np.arange(image.shape[0]))
    return ( pro.annulus(image, cx, cy, rad_in, rad_out), spots)

def define_control_halfannulus(image, cx= None, cy=None,
                               rad_in=None, rad_out=None,
                               theta2=None, theta1=None):
    spots = [(cx +rad_in, cy), (cx+rad_out, cy)]
    xs, ys = np.meshgrid( np.arange(image.shape[1]),
                            np.arange(image.shape[0]))
    return (pro.annuluswedge(image, cx, cy, rad_in, rad_out, theta2, theta1), spots)

if __name__ == "__main__":
    soft_ini  = 'Config/SN_Software.ini'
    soft_spec = 'Config/SN_Software.spec'
    config = flh.validate_configfile(soft_ini, soft_spec)
    #bgds = flh.setup_bgd_dict(config)

    centx = config['IM_PARAMS']['centerx']
    centy = config['IM_PARAMS']['centery']
    lambdaoverd = config['IM_PARAMS']['lambdaoverd']

    regionfilename = config['CONTROLREGION']['filename']
    innerlam = config['CONTROLREGION']['innerannulus']
    outerlam = config['CONTROLREGION']['outerannulus']
    print("Retrieving bgd, flat, badpix")
    bgds = flh.setup_bgd_dict(config)
    image = bgds['masterflat']
    ann, verts = define_control_halfannulus(image, cx = centx, cy = centy,
                         rad_in = lambdaoverd*innerlam,
                         rad_out= lambdaoverd*outerlam,
                         theta2 = 90, theta1=270)
    hdu = fits.PrimaryHDU(ann*1.0)
    hdu.writeto('controlregion.fits',overwrite = True)

    #flh.writeout(ann*1.0, regionfilename)
    config['CONTROLREGION']['verticesx'] = [centx]+[x[0] for x in verts]
    config['CONTROLREGION']['verticesy'] = [centy]+[y[1] for y in verts]
    config.write()
    print("Configuration file written to "+config.filename)
    plt.imshow(np.log(np.abs(ann*image)), origin='lower', interpolation = 'nearest')
    plt.show()
