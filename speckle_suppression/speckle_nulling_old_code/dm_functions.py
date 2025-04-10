from __future__ import division
from validate import Validator
import matplotlib.pyplot as plt
import sys
import numpy as np 
from configobj import ConfigObj
import numpy as np
import sn_filehandling as flh
from PIL import Image
from PIL import ImageDraw

def annulus(image, cx, cy, r1, r2):
    outer = circle(image, cx, cy, r2)
    inner = circle(image, cx, cy, r1)
    return ( outer-inner)

def circle(image, cx, cy, rad):
    zeroim = np.zeros(image.shape, dtype = np.int)
    for x in range(int(cx-rad), int(cx+rad+1)):
        for y in range(int(cy-rad), int(cy+rad+1) ):
            #print xs, ys
            dx = cx-x
            dy = cy -y
            if(dx*dx+dy*dy <= rad*rad):
                zeroim[y,x] = 1
    return zeroim

def rotateXY(xvals, yvals, thetadeg = 0):
    theta = np.pi/180.0*thetadeg
    return (np.cos(theta)*xvals- np.sin(theta)*yvals, 
            np.sin(theta)*xvals+ np.cos(theta)*yvals)

def intensitymodel( amp, k_rad, a=0, b=0, c=0):
    """Radial dependence of spot calibration\n
    intensity = amp**2*(a*k_rad**2 + b*k_rad + c)"""
    return  amp**2*(a*k_rad**2 + b*k_rad + c)

def amplitudemodel(counts, k_rad, a=0, b=0, c=0):
    """Radial dependence of spot calibration\n
    amplitude = sqrt(counts/(a*k_rad**2 + b*k_rad + c))""" 
    #fudge = 0.5
    fudge = 1
    retval = fudge*np.sqrt((counts/(a*k_rad**2 + b*k_rad + c)))
    if np.isnan(retval):
        return 0
    else:
        return retval

def text_to_flatmap(text, amplitude, x=15, y=15, N=21):
    image = Image.new('L', (N,N))
    draw = ImageDraw.Draw(image)
    draw.text((x, y), text, amplitude)
    j = np.asarray(image)
    return j

def make_speckle_kxy(kx, ky, amp, phase, N=21, flipy = True, flipx = False):
    """given an kx and ky wavevector, 
    generates a NxN flatmap that has 
    a speckle at that position"""
    dmx, dmy   = np.meshgrid( 
                    np.linspace(-0.5, 0.5, N),
                    np.linspace(-0.5, 0.5, N))
    xm=dmx*kx*2.0*np.pi
    ym=dmy*ky*2.0*np.pi
    
    fx = -1 if flipx else 1
    fy = -1 if flipy else 1
    ret = amp*np.cos(fx*xm + fy*ym +  phase)
    return ret

def make_speckle_xy(xs, ys, amps, phases, 
                    centerx=None, centery=None, 
                    angle = None,
                    lambdaoverd= None):
    """given an x and y pixel position, 
    generates a NxN flatmap that has 
    a speckle at that position"""
    #convert first to wavevector space
    kxs, kys = convert_pixels_kvecs(xs, ys, 
                  centerx = centerx,
                  centery = centery,
                  angle = angle,
                  lambdaoverd = lambdaoverd)
    returnmap = make_speckle_kxy(kxs,kys,amps,phases)
    return returnmap

def convert_pixels_kvecs(pixelsx, pixelsy, 
                    centerx=None, centery=None, 
                    angle = None,
                    lambdaoverd= None):
    """converts pixel space to wavevector space"""
    offsetx = pixelsx - centerx
    offsety = pixelsy - centery

    rxs, rys = rotateXY(offsetx, offsety, 
                            thetadeg = -1.0*angle)
    kxs, kys = rxs/lambdaoverd, rys/lambdaoverd
    return kxs, kys
                     
def convert_kvecs_pixels(kx, ky, 
                    centerx=None, centery=None, 
                    angle = None,
                    lambdaoverd= None):
    """converts wavevector space to pixel space"""
    rxs, rxy = kx*lambdaoverd, ky*lambdaoverd
    offsetx, offsety = rotateXY(rxs, rxy, 
                                    thetadeg = angle)
    pixelsx = offsetx + centerx
    pixelsy = offsety + centery
    return pixelsx, pixelsy

def annularmask(N, inner, outer):
    a = np.zeros((N,N))
    ret = annulus(a, float(N)/2-0.5, float(N)/2-0.5, inner, outer)
    return ret

def circularmask(N,rad):
    a = np.zeros((N,N))
    ret = circle(a, float(N)/2,float(N)/2, rad)
    return ret

if __name__ == "__main__":
    N=21
    fake_flatmap = np.zeros((N,N))
    configfilename = 'speckle_null_config.ini'
    configspecfile = 'speckle_null_config.spec'
    configspec = ConfigObj(configspecfile, _inspec = True)
    config = ConfigObj(configfilename, configspec= configspec)
    val = Validator()
    test = config.validate(val)
    
    centerx = config['IM_PARAMS']['centerx']
    centery = config['IM_PARAMS']['centery']
    angle = config['IM_PARAMS']['angle']
    lambdaoverd =config['IM_PARAMS']['lambdaoverd']
    dm = config['AOSYS']['dmcyclesperap']
    abc = config['INTENSITY_CAL']['abc']

#    flh.ds9(b)

