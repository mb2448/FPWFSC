import numpy as np

def rotateXY(xvals, yvals, thetadeg = 0):
    theta = np.pi/180.0*thetadeg
    return (np.cos(theta)*xvals - np.sin(theta)*yvals, 
            np.sin(theta)*xvals + np.cos(theta)*yvals)

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

def make_speckle_kxy(kx, ky, amp, phase, N=21, flipy = True, flipx = False, dm_rotation=0):
    """given an kx and ky wavevector, 
    generates a NxN flatmap that has 
    a speckle at that position
    
    Parameters
    ----------
    kx : float or ndarray
        x-component of the wavevector. If ndarray, must be same shape as ky 
        and output appends a dimension of size kx.shape[0]
    ky : float or ndarray
        y-component of the wavevector. If ndarray, must be same shape as kx
    amp: float 
        amplitude in physical units of meters
    phase: float
        phase in radians
    dm_rotation : float
        rotation of the DM about the propagation axis, degrees
    """

    
    dmx, dmy   = np.meshgrid( 
                    np.linspace(-0.5, 0.5, N),
                    np.linspace(-0.5, 0.5, N))
    
    if hasattr(kx, "shape") > 0:
        amp = np.asarray(amp)[..., None]
        dmx = dmx[..., None]
        dmy = dmy[..., None]

    xm=dmx*kx*2.0*np.pi
    ym=dmy*ky*2.0*np.pi

    xm, ym = rotateXY(xm, ym, thetadeg=dm_rotation)
    
    fx = -1 if flipx else 1
    fy = -1 if flipy else 1
    ret = amp*np.cos(fx*xm + fy*ym +  phase)
    return ret

def make_speckle_xy(xs, ys, amps, phases, 
                    centerx=None, centery=None, 
                    angle = None,
                    lambdaoverd= None,
                    N=22,
                    dm_rotation=0):
    """given an x and y pixel position, 
    generates a NxN flatmap that has 
    a speckle at that position"""
    #convert first to wavevector space
    kxs, kys = convert_pixels_kvecs(xs, ys, 
                  centerx = centerx,
                  centery = centery,
                  angle = angle,
                  lambdaoverd = lambdaoverd)
    returnmap = make_speckle_kxy(kxs,kys,amps,phases,N=N, dm_rotation=dm_rotation)
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

if __name__ == "__main__":
    print("todo")
    
    
    