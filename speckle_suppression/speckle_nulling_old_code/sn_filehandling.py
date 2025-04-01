############################## Import Libraries ###############################

## Math Library
import numpy as np
## Import OS information
import os
## import system library
import sys
## import time library
import time
## Import epics library. We mainly use the PV function
#from epics import *
## Import library used to manipulate .fits file
from astropy.io import fits
import astropy.io.fits as pf
## Imports libraries to manipulate configuration files
from configobj import ConfigObj
from configobj import flatten_errors
## Configuration Object Validator Library
from validate  import Validator
import warnings
import copy
import ipdb
from distutils.dir_util import copy_tree
from shutil import copyfile
## Are these functions used????
#import glob
# import configobj as co
#import time
#import datetime


class DataSaver:
    def __init__(self, defaultim=None, config = None):
        self.config = config
        self.defaultim = defaultim
        self.iterations = self.config['NULLING']['N_iterations']
        self.n_phases = len(self.config['NULLING']['phases'])

    def initialize(self):
        """Make all the directories and get things in order"""
        self.tstamp = time.strftime("%Y%m%d-%H%M%S").replace(' ', '_')
        self.outputdir = os.path.join(self.config['NULLING']['outputdir'],
                                      self.tstamp)
        if not os.path.exists(self.outputdir):
            print('Creating ',self.outputdir)
            os.makedirs(self.outputdir)
        #write config file to directory
        save_config = copy.copy(self.config)
        save_config.filename = os.path.join(self.outputdir, 'nulling_config.ini')
        save_config.write()
        #make subdirectories for each iteration
        for i in range(self.iterations):
            it_dir = self.get_itdir(i)
            if not os.path.exists(it_dir):
                os.makedirs(it_dir)
        #save bgds FIX THIS
        try:
            print("Saving detector cal files to output directory")
            copy_tree(self.config['DETECTOR_CAL']['dirwr'], self.outputdir)
        except:
            print("Directory not created, bgds not copied")
        try:
            print("Saving controlregion")
            inputfile = self.config['CONTROLREGION']['filename']
            outputfile = os.path.join(self.outputdir, 'controlregion.fits')
            copyfile(inputfile, outputfile)
        except:
            print("Control region not saved, some error")
        return

    def save_textfile(self, string, it=None, name=None):
        dirpath = self.get_itdir(it)
        fname = name
        with open(os.path.join(dirpath, fname), "w") as text_file:
            text_file.write(string)
        return

    def write_apertureim(self, speckle_aps=None, it=None):
        """An image of the speckle apertures, binary"""
        dirpath = self.get_itdir(it)
        fname = 'speckle_aps.fits'
        writeout(speckle_aps, outputfile=os.path.join(dirpath, fname))
        return

    def write_image(self, clean=None, raw=None, it=None,
                    offset=None, intensity=None):
        """Write the main image of each loop"""
        dirpath = self.get_itdir(it)
        comment = str(offset)+', '+str(intensity)
        if clean is not None:
            fname = 'image_clean.fits'
            writeout(clean, outputfile=os.path.join(dirpath, fname))
        if raw is not None:
            fname = 'image_raw.fits'
            writeout(raw, outputfile=os.path.join(dirpath, fname))
        return

    def write_phaseim(self, clean=None, raw=None, it=None, phase_it=None,
                      offset=None, intensity=None):
        """writes to the phase directory the phase images and puts the offset
        in the header under COMMENT"""
        dirpath = self.get_itdir(it)
        comment = str(offset)+', '+str(intensity)
        if clean is not None:
            fname = 'phase_'+str(phase_it)+'_clean.fits'
            writeout(clean, outputfile=os.path.join(dirpath, fname),
                     comment=comment)
        if raw is not None:
            fname = 'phase_'+str(phase_it)+'_raw.fits'
            writeout(raw, outputfile=os.path.join(dirpath, fname),
                     comment=comment)
        return

    def write_dm_shape(self, data, it=None, name=None):
        dirpath = self.get_itdir(it)
        fname = 'dm_'+name+'.fits'
        writeout(data, outputfile=os.path.join(dirpath, fname))
        return

    def get_itdir(self, it):
        """Returns the subdirectory corresponding to a particular iteration"""
        return os.path.join(self.outputdir, 'iteration'+str(it))

class Output_imagecube:
    def __init__(self, n, defaultim, filepath = None, comment = None, configfile = None):
        self.size_x = defaultim.shape[0]
        self.size_y = defaultim.shape[1]
        self.cube = np.zeros( (n,self.size_x, self.size_y))
        writeout(self.cube, outputfile = filepath,
                            comment =comment)
        self.i = 0
        self.filepath = filepath

    def config_to_string(self, configfile):
        stringy = ''
        with open(configfile) as f:
            for line in f:
                stringy = stringy+line
        return stringy

    def update(self, array ):
        self.cube[self.i, :,:] = array
        self.i = self.i+1
        writeout(self.cube, outputfile = self.filepath)

# =============================================================================
def validate_configfile(ini, spec):
    ''' -----------------------------------------------------------------------
    ----------------------------------------------------------------------- '''
    # Parse config file provided by yser
    config = ConfigObj( ini, configspec = spec)
    # Instancie the validator class
    val = Validator()
    # Check if the config file match the requirement of the spec file
    res = config.validate(val, preserve_errors = True)
    # If the config file pass the validor test, return the config file
    if res is True: return config
    # If the config file failed to pass the validator test:
    else:
        # Prepare message for the user
        msg  = "Warning: "
        msg += "the config file does not match spec file requierements."
        # Print warning message for the user
        print(msg)
        # Print name of all problematic items.
        for item in flatten_errors(config, res): print(item)
        # Raise a Value Error
        raise ValueError('Configuration file corrupted.')

# =============================================================================
def setup_bgd_dict(config):
    ''' -----------------------------------------------------------------------
    Creates a dictionary of 'bkgd', 'masterflat', 'badpix' with the correct
    images as the values.  this makes for easy 'dereferencing' when using
    equalize_image(image, **bgd)
    ----------------------------------------------------------------------- '''
    # Get the directory where the Detector calibration data has been saved
    bgddir = config['DETECTOR_CAL']['dirwr']
    # Read the calibration data
    # bkgd       = fits.open(os.path.join(bgddir, 'medbackground.fits'))[0].data
    bkgd = np.zeros([264,256])
    # masterflat = fits.open(os.path.join(bgddir, 'masterflat.fits'   ))[0].data
    masterflat = np.ones([264,256])
    # badpix     = fits.open(os.path.join(bgddir, 'badpix.fits'       ))[0].data
    badpix  = np.zeros([264,256])
    # Build the dictionary
    dictionary = { 'bkgd': bkgd, 'masterflat': masterflat, 'badpix' : badpix }
    # Return the dictionary
    return dictionary

class Printer():
    """
    Print things to stdout on one line dynamically
    """
    def __init__(self,data):

        sys.stdout.write("\r\x1b[K"+data.__str__())
        sys.stdout.flush()

class Timer():
    """
    Time something like a for loop
    only argument is max_iterations
    Ex:
    tt = Timer(100)
    for i in range(100):
        Printer(tt.timeleft())
    """
    def __init__(self, imax):
        self.t0 = time.time()
        self.i  = 0
        self.imax = imax
    def timeleft(self):
        telapsed =time.time()-self.t0
        pctdone = 100.0*(self.i+1)/self.imax
        tleft = (1-pctdone/100)/(pctdone/100/telapsed)
        percentstr= (format(pctdone, '.2f')+
                       "% done.  ")
        if tleft > 120:
            timestr = str(datetime.timedelta(
                            seconds=int(tleft)))
        else:
            timestr   = (format(tleft, '.1f')+" seconds")
        self.i = self.i+1
        return ("  "+percentstr+timestr + " remaining")

    def timeelapsed(self):
        timestr = "  "+(format(time.time-self.t0, '.1f'))+" time  elapsed"
        return timestr

def parsenums(linestring):
    """converts strings like '1-10; 15-17' into a list of ph0001.fits, ph0002.fits, etc"""
    ans = []
    first=linestring.split(';')
    for thing in first:
        ans= ans + (range(
                    int(thing.split('-')[0]),
                    int(thing.split('-')[1])+1))

    return ['ph'+str(x).zfill(4)+'.fits' for x in ans]

def int_if_possible(value):
    try: return int(value)
    except: return value

def float_if_possible(value):
    try: return float(value)
    except: return value

def intdict(dicty):
    """Converts values in dictionary to integers"""
    return dict((k, int_if_possible(v)) for (k, v) in dicty.items())

def floatdict(dicty):
    """Converts values in dictionary to floats"""
    return dict((k, float_if_possible(v)) for (k, v) in dicty.items())

def check_exptime(filelist, t=1416):
    outputlist = []
    for fitsfile in filelist:
        hdulist= pf.open(fitsfile)
        header = hdulist[0].header
        #replace this with a global check of all parameters
        if header['T_INT'] != t:
            print ("\nWarning: "+fitsfile+
               " has different exposure time of "+
               str(header['T_INT'])+
               " instead of "+str(t)+
               ", skipping it")
        else:
            outputlist.append(fitsfile)
        hdulist.close()
    return outputlist

def check_equal(iterator):
    #checks if everything in list, etc is equal
    return len(set(iterator)) <= 1

def sameheadervals(db):
    """returns a list of header keys which
    all have the same values in the db"""
    keys = db.keys()
    passedkeys=[]
    for x in keys:
        if validate(db,x):
           passedkeys.append(x)
    return passedkeys

def diffheadervals(db):
    """returns a list of header keys which
    have the different values in the db"""
    keys = db.keys()
    passedkeys=[]
    for x in keys:
        if not validate(db,x):
           passedkeys.append(x)
    return passedkeys

def stripheader(header, keys):
    headercopy = header.copy()
    for key in keys:
        headercopy[key]=-99999
    return headercopy

def validate(databasesubset, fieldstocheck=None):
    """>>>validate(db[db['Type']=='flats'], ['FILTER','T_INT'])
    checks to see if all files of type flats in the db have the
    same t_int and filter"""
    if not isinstance(fieldstocheck,list):
        fieldstocheck=[fieldstocheck]
    passed = True
    for field in fieldstocheck:
        if check_equal(databasesubset[field]):
            pass
        else:
            passed = False
            print( "WARNING. NOT ALL "+field+" THE SAME")
            #print databasesubset[field]
    return passed

def ds9(data):
    writeout(data, 'temp.fits')
    os.system('/Applications/ds9/ds9 temp.fits &')
    pass

def get_paths_dir(directory, numstr):
    filelist = parsenums(numstr)
    returndict=[os.path.join(directory, x) for x in filelist]
    return returndict

def formattoarray(string):
    t1 = string.replace('[', '').replace(']', '').split(' ')
    l1=[]
    for x in t1:
        try:
            l1.append(float(x))
        except:
            pass
    return np.array(l1)

def formatnum(string):
    temp="".join([x if x in ['1','2','3','4','5','6','7','8','9','0','.', '-'] else '' for x in string])
    return temp

def dictfromfile(filename, **kwargs):
    """Usage: dict = dictfromfile("myfile.txt", delim=" ")"""
    lines= np.genfromtxt(filename, **kwargs)
    dicty = {}
    for i in range(len(lines[0,:])):
        try:
            dicty[str(lines[0,i]).strip(" ")]=np.array(
                lines[1:, i], dtype=float)
        except:
            dicty[str(lines[0,i]).strip(" ")]=np.array(
                lines[1:, i], dtype=str)
    return dicty

def get_paths_conf(configobj, directory = None):
    returndict={}
    """returns a dict of {'targ':[/users/me/ph0001.fits,...], 'cal':[/users/me/ph0003.fits"""
    for item in configobj['Dirs']['Input'].keys():
        try:
            filelist = parsenums(configobj['Dirs']['InputFileNums'][item])
            if directory is not None:
                returndict[item]=[os.path.join(
                   directory,  x) for x in filelist]
            else:
                returndict[item]=[os.path.join(
                    configobj['Dirs']['Input'][item], x) for x in filelist]
        except:
            print ("Warning! "+item+" SKIPPED")
    return returndict

def write2columnfile(A, B,
                    filename = 'twocols.txt',
                    header = None):
    with open(filename,  'w') as f:
        if header is not None:
            print >>f, header
        for f1, f2 in zip(A, B):
            print >> f, f1, f2
    pass

def writeout(data, outputfile, header=None, comment=None):
    pf.writeto(outputfile, data, header=header, overwrite=True)
    writtenfile = pf.open(outputfile)
    if comment is not None:
        writtenfile[0].header.set('COMMENT', comment)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        writtenfile.writeto(outputfile, output_verify='ignore', clobber=True)
        writtenfile.close()

def find_all_fits(rootdir):
    files=[]
    for dirpath,_,filenames in os.walk(rootdir):
        for f in filenames:
            if f.endswith('.fits'):
                files.append(os.path.abspath(os.path.join(dirpath, f)))
    return files

def get_latest_file(filetype = None, directory = None):
    searchstring = os.path.join(directory, '*'+filetype)
    return max(glob.iglob(searchstring), key=os.path.getctime)

def get_latest_fitsfile(directory):
    return get_latest_file(directory =directory, filetype='fits')
