import numpy as np
import configobj as co
import time
import sn_filehandling as flh
import sn_processing as pro
import sn_preprocessing as pre
import sn_math as snm
import os
import astropy.io.fits as pf
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as sciim
pd.set_option('display.width', 1000)
pd.options.display.max_colwidth=1000

def contrastcurve_simple(image, cx=None, cy = None, 
                         fwhm = 1, sigmalevel = 1, robust=True,
                         region =None, maxrad = None):
    if cx is None:
        cx = image.shape[0]//2
    if cy is None:
        cy = image.shape[0]//2
    
    if maxrad is None:
        maxpix = image.shape[0]//2
    else:
        maxpix = maxrad
    pixrad = np.arange(maxpix)
    clevel = pixrad*0.0
    if region is None:
        region = np.ones(image.shape)
    for idx, r in enumerate(pixrad):
        annulusmask = pro.annulus(image, cx, cy, r, r+fwhm)
        if robust:
            sigma = snm.robust_sigma(image[np.where(np.logical_and(annulusmask, region))].ravel())
        else:
            sigma = np.std(image[np.where(np.logical_and(annulusmask, region))].ravel())
        
        clevel[idx]=sigmalevel*(sigma)
        flh.Printer(str(idx) + ' '+str( clevel[idx]))
    return (pixrad, clevel)

def generate_curveimage(image, pixrad, clevel, cx=None, cy = None):
    """Returns an image where the intensity at each point corresponds
        to the contrast level you pass as clevel"""
    if cx is None:
        cx = image.shape[0]//2
    if cy is None:
        cy = image.shape[0]//2
    outim = np.ones(image.shape)
    for idx, r in enumerate(pixrad):
        annulusmask = pro.annulus(image, cx, cy, r, r+1)
        outim[np.where(annulusmask)]=clevel[idx]
    return outim


def contrastcurve(image, psf, cx = None, cy = None, 
                  plsc = .025, fwhm = 3.4, kernel = None,
                  sigmalevel=5, conv=False, robust = False,
                  max_fov=6, filtermultfactor = 36.0):
    """compute the contrast curve from a reduced image
       and a psf image.  
        image = image to compute curve for
        cx, cy = center x and center y of image (pixels)
        psf   = psf image for contrast zeropoint
        sigmalevel = 5
        conv = T/F whether to perform a gaussian convolution
                    on the image
        fwhm = fwhm of gaussian to convolve AND used to compute 
               annular width in contrast curve
        plsc = platescale (arc-seconds/pixel)
        max_fov = maximum field of view in arc seconds
        robust  = use a robust sigma measure
        multfactor = any multiplication factor resulting from
                     different filters or grisms
    """

    if (cx is None) or (cy is None):
        cx, cy = pre.get_spot_locations(image, comment='click on the center pixel')[0]
    if conv:
        print( "Convolving PSF with kernel")
        convim = pro.matchfilter(image, kernel)
        print( "Fitting PSF")
        psfamp=pre.quick2dgaussfit(pro.matchfilter(psf, kernel), xy = [cx, cy])[0]
    else:
        convim = image
        psfamp = pre.quick2dgaussfit(psf, xy = [cx, cy])[0] 
    
    maxpix = min(int(max_fov/plsc), image.shape) 
    pixrad = np.arange(maxpix)
    clevel = np.arange(maxpix)*0.0

    #plt.ion()
    #fig, ax0 = plt.subplots(ncols=1, figsize = (8, 8))
    #ax=plt.imshow(convim, interpolation='nearest',origin='lower')
    #plt.show()
    time.sleep(1)
    for idx, r in enumerate(pixrad):
        annulusmask = pro.annulus(convim, cx, cy, r, r+fwhm)
        #sigma = rs.robust_sigma(convim[np.where(annulusmask)].ravel())
        if idx <1:
            sigma = None
            clevel[idx]==None
            continue
        if robust is True:
            sigma = plm.robust_std(convim[np.where(annulusmask)].ravel())
        else:
            sigma = np.std(convim[np.where(annulusmask)].ravel())
        
        clevel[idx]=sigmalevel*(sigma)/psfamp/filtermultfactor
        printout = [str(x) + ', ' for x in [r,pixrad[idx]*plsc,clevel[idx] ]]
        flh.Printer( ''.join(printout))
        #ax.set_data(convim*annulusmask)
        #plt.draw()
    #plt.close()
    #plt.ioff()
    return (pixrad*plsc, clevel)

def rawcontrastcurve(image, psf=None, cx = None, cy = None, 
                  fwhm = 3.4, plsc = .025,
                  sigmalevel=5, gaussfilt=False,robust = False, 
                  max_fov=6, multfactor = 36.0, rawmultfactor = 1.0):
    if cx is None or cy is None:
        #cx, cy = pre.quickcentroid(image)
        cx, cy = 256, 256
    maxpix = min(int(max_fov/plsc), image.shape) 
    if gaussfilt:
        kernel = plm.gausspsf2d(10, fwhm)
        convim = sciim.filters.convolve(image, kernel)
    else:
        convim = image

    if psf is not None:
        psfamp=pre.quick2dgaussfit(psf)[0]
    
    pixrad = np.arange(maxpix)
    clevel = np.arange(maxpix)*0.0

    for idx, r in enumerate(pixrad):
        annulusmask = pro.annulus(convim, cx, cy, r, r+fwhm)
        if idx <1:
            sigma = None
            clevel[idx]==None
            continue
        if robust == True:
            sigma = plm.robust_std(convim[np.where(annulusmask)].ravel())
        else:
            sigma = np.std(convim[np.where(annulusmask)].ravel())
        if psf is not None:
            clevel[idx]=sigmalevel*(sigma)/psfamp/multfactor
        else:
            clevel[idx]=sigmalevel*sigma/rawmultfactor
        flh.Printer( str(idx)+", "+ str(pixrad[idx]*plsc)+", "+ str(clevel[idx]))
        
    return (pixrad*plsc, clevel)


#if __name__=="__main__":
#    
#    configfile = ('/Users/Me/Desktop/research/Coronagraphy/'+
#                 'SDC_Pipeline/reduction_configuration_files/'+
#                 'delandreduction.ini')

#def run(configfile):
#    conf = co.ConfigObj(configfile)
#    basedir = os.path.join(
#         conf['BASEOUTDIR'],
#         conf['TARGNAME'],
#         conf['NAME'])
#    
#    cubedir = os.path.join(basedir, 
#                    conf['Processing']
#                        ['Outputdirs']
#                        ['cubedir']) 
#    
#    subtractdir = os.path.join(basedir, 
#                    conf['Processing']
#                        ['Outputdirs']
#                        ['subtractdir']) 
#
#    pcadir= os.path.join(basedir,
#                     conf['PCA']
#                         ['Outputdirs']
#                         ['pcadir']) 
#   
#    patchpcadir = os.path.join(basedir,
#                    conf['PATCH_PCA']
#                        ['Outputdirs']
#                        ['patchpcadir'])
#
#    psfdir = os.path.join(basedir, 
#                    conf['Processing']
#                        ['Outputdirs']
#                        ['psfdir']) 
#    
#    psffile = pf.open(
#             os.path.join(psfdir, 'medianpsf.fits'))
#    
#    subtracttitlestring = os.path.join(subtractdir, 'subtracted.fits')
#    
#    if 'supercube' in conf['PATCH_PCA']['Setup'].keys():
#        if conf['PATCH_PCA']['Setup']['supercube'] == 'True':
#            supercube = True
#            supercubeflag = '_supercube_'
#        else:
#            supercube = False
#            supercubeflag = ''
#    else:
#        supercube = False
#        supercubeflag = ''
#    
#    patchcubeconfig = flh.intdict(conf['PATCH_PCA']['Setup'])
#    patchpcatitlestring = os.path.join(
#        patchpcadir,
#        ('patchpca_'+
#        supercubeflag+
#        str(patchcubeconfig['seglen'])+'x'+
#        str(patchcubeconfig['segwidth'])+'_ncomps_'+
#        str(patchcubeconfig['n_components'])+'_step_'+
#        str(patchcubeconfig['step'])+
#        '.fits'))
#    if 'supercube' in conf['PCA']['Method']:
#        if conf['PCA']['Method']['supercube'] == 'True':
#            supercube = True
#            supercubeflag = '_supercube_'
#        else:
#            supercube = False
#            supercubeflag = ''
#    else:
#        supercube = False
#        supercubeflag = ''
#
#    pcatitlestring = os.path.join(pcadir, 
#                   ('pca_mediancube'+'_comps_'+
#                   conf['PCA']['Params']['n_components']+
#                   supercubeflag+
#                   '.fits'))
#    if 'PATCH_ANNULUS' in conf.keys():
#        if conf['PATCH_ANNULUS']['Setup']['supercube'] == 'True':
#            supercube = True
#            supercubeflag = '_supercube_'
#        else:
#            supercube = False
#            supercubeflag = ''
#        try:
#            annconfig= flh.intdict(conf['PATCH_ANNULUS']['Setup'])
#            annularpcatitlestring = os.path.join(patchpcadir,
#                                        ('patchpca_annulus_'+
#                                        supercubeflag+
#                                        str(annconfig['rmin'])+'-'+
#                                        str(annconfig['rmax'])+'-'+
#                                        str(annconfig['deltarad'])+
#                                        '_annwidth_'+str(annconfig['annwidth'])+
#                                        '_thetastep_'+str(annconfig['thetastep'])+
#                                        '_thetawidth_'+
#                                        str(annconfig['thetawidth'])+
#                                        '_ncomps_'+str(annconfig['n_components'])+
#                                        '_autocomps_'+
#                                        str(annconfig['autocomponents'])+
#                                        '_compdens_'+
#                                        str(annconfig['componentdensity'])+
#                                        '.fits'))
#        except:
#            annularpcatitlestring = None
#    else:
#        annularpcatitlestring = None
#   
#    rawimageheader = pf.open(
#                    os.path.join(cubedir, 'targcube_recentered.fits'))[0].header
#    psf, psfheader   = psffile[0].data, psffile[0].header
#    psfkernel = pf.open(
#                 os.path.join(psfdir, 'psfkernel.fits'))[0].data
#    
#    expratio = float(rawimageheader['T_INT'])/psfheader['T_INT']
#    filterratio = ( pro.pharofilterstrength(rawimageheader['FILTER'])/
#                    pro.pharofilterstrength(psfheader['FILTER']) )
#    grismratio = ( pro.pharogrismstrength(rawimageheader['GRISM'])/
#                    pro.pharogrismstrength(psfheader['GRISM']) )
#    totalratio = expratio*filterratio*grismratio
#
#    for imagetitle in [pcatitlestring, patchpcatitlestring, subtracttitlestring, annularpcatitlestring]:
#        if imagetitle is None:
#            continue
#        print( "Computing contrast for "+str(imagetitle))
#
#        imagefile = pf.open(imagetitle)
#        image, imageheader = imagefile[0].data, imagefile[0].header
#
#        cx, cy = image.shape[0]//2, image.shape[1]//2
#        angsep, contrast = contrastcurve(image, cx = cx, cy= cy, psf=psf, max_fov=4,  
#                                        filtermultfactor = totalratio , conv = True,
#                                        kernel=psfkernel)
#        contrast = np.array(contrast)
#        #to generate an SNRimage
#
#        pixrad, simplecontrast = contrastcurve_simple(image, cx=cx, cy=cy)
#        contcurveimage = generate_curveimage(image, pixrad,
#                                             simplecontrast, cx=cx, cy=cy) 
#        plt.semilogy(angsep, contrast, 'k.')
#        plt.ylim((10**(-6), 10**(-0)))
#        plt.xlim((0,6))
#        plt.title(imagetitle.split('/')[-1]+' Contrast')
#        plt.xlabel('Arcseconds')
#        plt.ylabel('Contrast')
#        plt.savefig(imagetitle+'_contrastcurve.pdf', dpi = 300)
#        plt.close()
#        
#        with open(imagetitle+'_contrastcurvedata.txt', 'w') as f:
#            for f1, f2, in zip(angsep, contrast):
#                print(f, f1, f2)
#        
#        print '\n'
#        flh.writeout(image/contcurveimage, imagetitle+'_snrimage.fits')
