############################## Import Libraries ###############################

## Math Library
import numpy as np
## import system library
import sys

# Location of the FIU libraries
sys.path.append('/kroot/src/kss/nirspec/nsfiu/dev/lib/')
## Library used to control NIRC2
import Nirc2_cmds as Nirc2
## Library used to control Keck II AO
import K2AO_cmds as K2AO

# Location of the Speckle Nulling libraries
sys.path.append('/kroot/src/kss/nirspec/nsfiu/dev/speckle_nulling/')
## Library to handle speckle nulling files
import sn_filehandling  as flh 


##### =========================================================================
class Fake_AO_System():
    ''' -----------------------------------------------------------------------
    ----------------------------------------------------------------------- '''

    # =========================================================================
    def __init__(self, name, ini = None, spec = None):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        # Update the local name of the configuration section associated to this
        # class
        self.name = name        
        # Read the config file and check if all parameters are valid        
        classconfig = flh.validate_configfile(ini, spec)
        # Creates a local copy of the class config
        self.classconfig = classconfig[self.name]
        # Create local variable containing DM size (X and Y)
        #self.dm_size_x = self.classconfig[]
        #self.dm_size_y = self.classconfig[]
        # This function does not return anything
        return 

    # =========================================================================
    def get_dm_shape(self):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        # Get the shape currently apply to the Keck II AO DM
        tmp = self.K2AO.get_DM_shape()
        # Return the shape of the DM read at the previous step
        return self.K2AO.DM_shape_map

    # =========================================================================    
    def set_dm_shape(self, shape):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        # Check size of the DM shape provided
        if np.shape(shape) == (self.dm_size_x,self.dm_size_y):            
            # Return True is DM map provided has the correct shape
            return True 
        else:
            # Return False is DM map provided has the not correct shape
            return False


##### =========================================================================
class KECK2AO:
    ''' -----------------------------------------------------------------------
    ----------------------------------------------------------------------- '''

    # =========================================================================
    def __init__(self, name, ini = None, spec = None):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        # Update the local name of the configuration section associated to this
        # class
        self.name = name        
        # Read the config file and check if all parameters are valid        
        classconfig = flh.validate_configfile(ini, spec)
        # Create local variable containing DM size (X and Y)
        #self.dm_size_x = self.classconfig[]
        #self.dm_size_y = self.classconfig[]
        # Instancie Keck 2 AO class
        self.K2AO = K2AO.K2AO_cmds()
        # Do not return anything
        return 

    # =========================================================================
    def get_dm_shape(self):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        # Get the shape currently apply to the Keck II AO DM
        tmp = self.K2AO.get_DM_shape()
        # Return the shape of the DM read at the previous step
        return self.K2AO.DM_shape_map

    # =========================================================================    
    def set_dm_shape(self, shape):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        # Apply the shape provided to the Keck II AO DM
        tmp = self.K2AO.set_DM_shape(shape, style = 'abs')
        # Return True is shape applied properly. Return False otherwise 
        return tmp 

    # =========================================================================    
    def save_dm_shape(self,directory = '', filename = ''):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        # Save current Keck II AO shape
        tmp = self.K2AO.save_dm_shape(dirwr = directory, filename = filename)
        # Returns location where dm shape has been saved
        return tmp 


    # =========================================================================    
    def load_dm_shape(self,filename):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        # Save current Keck II AO shape
        tmp = self.K2AO.load_dm_shape(filename = filename)
        # Returns True if map load and apply properly. Returns False otherwise.
        return tmp 
    
    def TTM_move(self, move_amt):
        """move_amt should be the amoutn you move in x and y"""
        tmp = self.K2AO.TTM_move(move_amt)
        return tmp
##### =========================================================================
class Fake_Detector():
    ''' -----------------------------------------------------------------------
    This class simulate a fake detector. 
    Returns a random image plus some random noise.
    ----------------------------------------------------------------------- '''

    # =========================================================================
    def __init__(self, name, ini = None, spec = None):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        # Update the local name of the configuration section associated to this
        # class
        self.name = name        
        # Read the config file and check if all parameters are valid        
        classconfig = flh.validate_configfile(ini, spec)
        # Creates a local copy of the class config
        self.classconfig = classconfig[self.name]
        # Creat local image size parameters (x and y). Update these values 
        #self.im_size_x = self.classconfig[]
        #self.im_size_y = self.classconfig[]
        #self.verbose   = self.classconfig[]
        # Specific Parameters
        #self.iteration = self.classconfig[]
        #self.med_im    = self.classconfig[]
        #self.rand_amp  = self.classconfig[]
        # Do not return anything
        return

    # =========================================================================
    def take_image(self):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        # Print a warning message if verbse is True
        if self.verbose:
            print("Warning, you are using fake pharo simulator.")
        # Update the iteration number
        self.iteration += 1
        # Generate a see for the random function
        seed = np.random.randint(0, 28)
        # Generate a fake image
        fake_im  = np.random.random((self.im_size_x, self.im_size_y))
        fake_im *= self.rand_amp
        fake_im += self.med_im
        # Return fake images
        return fake_im

##### =========================================================================
class NIRC2:
    ''' -----------------------------------------------------------------------
    ----------------------------------------------------------------------- '''
    
    # =========================================================================
    def __init__(self, name, ini = None, spec = None):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        # Update the local name of the configuration section associated to this
        # class
        self.name = name
        # Read the config file and check if all parameters are valid
        config = flh.validate_configfile(ini, spec)
        # Creates a local copy of the class config
        self.config = config[self.name]
        # Instancie Nirc2 class. 
        # In this case it also pull all Nirc2 parameters.
        self.Nirc2  = Nirc2.Nirc2_cmds()
        # Define the size of the image
        self.im_size_x = self.Nirc2.x_extent
        self.im_size_y = self.Nirc2.y_extent
        # This function does not return anything
        return 

    # =========================================================================    
    def take_image(self,):
        ''' -------------------------------------------------------------------
        Returns an image in the default camera configuration
        ------------------------------------------------------------------- '''
        return self.Nirc2.take_image()[0]

    # =========================================================================
    def print_parameters(self,):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        # Print all Nirc2 parameters
        self.Nirc2.print_all_parameters()
        # This function does not return anything
        return 

    # =========================================================================
    def update_parameters(self,):
        ''' -------------------------------------------------------------------
        ------------------------------------------------------------------- '''
        # Update all Nirc2 parameters
        self.Nirc2.get_all_parameters()
        # Update image size parameters (x and y).  
        self.im_size_x = self.Nirc2.x_extent
        self.im_size_y = self.Nirc2.y_extent
        # This function does not return anything
        return 

