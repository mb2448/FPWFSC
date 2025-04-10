import time
import matplotlib.pyplot as plt 
#import ipdb
#import image_registration as imreg
import astropy.io.fits as pf
import dm_functions as DM
import configobj as co
import os
import sn_filehandling as flh
import sn_preprocessing as  pre
import numpy as np
import scipy.ndimage as sciim
import sys

def pharofilterstrength(filt):
    filtdict={'K_short' :1.0,
              'Br-gamma':0.08}
    return filtdict[filt]

def pharogrismstrength(grism):
    grismdict={'Open':1.0,
               'ND 1%': .02095,
               'ND 0.1%': .001710}
    return grismdict[grism]

def pharofilterratio(filt1, filt2):
    return pharofilterstrength(filt1)/pharofilterstrength(filt2)

def pharogrismratio(grism1, grism2):
    return pharogrismstrength(grism1)/pharogrismstrength(grism2)

def matchfilter(image, psf):
    return sciim.filters.convolve(image, psf[::-1, ::-1], mode='nearest')

def correlation_coeff(im1, im2):
    cc = (np.sum((im1-np.mean(im1)) * (im2-np.mean(im2))) / 
            np.sqrt( np.sum( (im1-np.mean(im1))**2 ) * 
                np.sum( (im2-np.mean(im2))**2 ) ))
    return cc

def rotate_cube(cube, vector, diff = False):
    if diff:
        diffvector = vector -vector[0]
    else:
        diffvector = vector
    newcube = np.zeros(cube.shape)
    for i in range(cube.shape[0]):
        flh.Printer(str(i))
        newcube[i,:,:] = sciim.interpolation.rotate(cube[i,:,:], 
                -1*diffvector[i], reshape = False, order = 1)
    
    return newcube

def disttocenter(image):
    cx = image.shape[0]//2
    cy = image.shape[1]//2
    
    ipdb.set_trace()
    if image.shape[0]%2 ==1:
        xs = np.arange(image.shape[0])-cx
    else:
        xs = np.arange(image.shape[0])-cx+0.5
    if image.shape[1]%2 ==1:
        ys = np.arange(image.shape[1])-cy
    else:
        ys = np.arange(image.shape[1])-cy+0.5
    xp, yp = np.meshgrid(xs, ys)
    d = np.sqrt(xp**2 + yp**2)
    return d

def rotate_image(image, angle):
    return sciim.interpolation.rotate(image, -1*angle, reshape=False, order=1)


def normalize_poisson_noise(datacube):
    """from Surf. Interface Anal. 2004; 36:203-212"""
    dc = datacube.reshape((datacube.shape[0], datacube.shape[1]*datacube.shape[2]))
    aG = np.mean(dc, axis = 1)
    bH = np.mean(dc, axis = 0)
    
    scaled = (1.0/np.sqrt(aG))[:,np.newaxis]*dc*(1.0/np.sqrt(bH))
    return np.nan_to_num(scaled.reshape(datacube.shape)), aG, bH


def denormalize_poisson_noise(normed_datacube, aG, bH):
    return np.sqrt(aG)[:, np.newaxis]*normed_datacube*np.sqrt(bH)

def topcorrelated(im1, cube, n):
    if n>= cube.shape[0]:
        print( "Warning: top N correlated greater than number of images")
        return cube
    else:
        ccs = np.zeros(cube.shape[0])
        for i in range(cube.shape[0]):
            ccs[i]=correlation_coeff(im1, cube[i,:,:])
        ccorder = np.argsort(ccs)[::-1]
    return cube[ccorder[0:n],:,:]

def mask(cube, mask):
    return cube*mask


def annulus(image, cx, cy, r1, r2):
    outer = circle(image, cx, cy, r2)
    inner = circle(image, cx, cy, r1)
    return ( outer-inner)

def circle_old(image, cx, cy, rad):
    x, y = np.meshgrid( np.arange(image.shape[1], dtype = np.float32),
                            np.arange(image.shape[0], dtype = np.float32))
    return (x-cx)**2+(y-cy)**2<=rad**2

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

def annuluswedge(image, cx, cy, rad1, rad2, theta2=360, theta1=0):
    outer= circlewedge(image, cx, cy, rad2, theta2, theta1)
    inner= circlewedge(image, cx, cy, rad1, theta2, theta1)
    return (outer - inner)

def circlewedge(image, cx, cy, rad, theta1=0, theta2=360):
    x, y = np.meshgrid( np.arange(image.shape[1]),
                            np.arange(image.shape[0]))
    reg1 = (x-cx)**2+(y-cy)**2<rad**2
    if ((theta2<=180 ) & (theta1>=0)):
        reg2 = np.angle((x-cx)+1.0j*(y-cy), deg=True)>=theta1
        reg3 = np.angle((x-cx)+1.0j*(y-cy), deg=True)<theta2
        return 1*np.logical_and(np.logical_and(reg1, reg2), reg3)
    if ((theta2>180)&(theta1<=180)):
        reg2 = np.angle((x-cx)+1.0j*(y-cy), deg=True)>=theta1
        reg3 = np.angle((x-cx)+1.0j*(y-cy), deg=True)<=theta2-360
        return 1*np.logical_and(np.logical_or(reg2, reg3), reg1)
    if ((theta2>180)&(theta1>=180)):
        reg2 = np.angle((x-cx)+1.0j*(y-cy), deg=True)>=theta1-360
        reg3 = np.angle((x-cx)+1.0j*(y-cy), deg=True)<=theta2-360
        return 1*np.logical_and(np.logical_and(reg2, reg3),reg1)
    else:
        print( "Please enter valid angles:")
        print( "syntax: def circlewedge(image, cx, cy, rad, theta2=360, theta1=0):")
        sys.exit(0)

if __name__ == "__main__":
    n_ints = 100
    testim = np.zeros((1024, 1024))
    test_int = np.random.randint(400, 512, (n_ints, 2))
    test_float = test_int+np.random.random(test_int.shape)
    for i in range(n_ints):
        x = test_float[i,0]
        y = test_float[i,1]
        rad = np.random.randint(1, 4)+np.random.random()
        #a = circle_old(testim, x, y, rad) 
        b = circle(testim, x, y, rad) 
        #print t1-t0
        if np.allclose(b,b):
            print( x, y, rad, " Pass")
        else:
            print( x, y, rad, " Fail")
