
import sys
sys.path.insert(0, '/usr/local/home/mcisse/PyAO/')
import ipdb
import warnings
import hcipy
import numpy as np
from aoscripts.ao_systems.k2ao import K2AO as KeckAO
from aosys.nirc2.nirc2 import Nirc2 as NIRC2
from aoscripts.reconstructor_manager.reconstructor_manager import ReconstructorManager
try:
    import numpy

    
    #from guis.fast_and_furious.hardware import NIRC2, OSIRIS, KeckAO
    #import aosys.xinetics_deformable_mirror as xd
    #from aosys.shwfs.shwfs import SHWFS
    #from aosys.shwfs_field_steering_mirror.shwfs_field_steering_mirror import SHWFSFieldSteeringMirror
    #from aosys.rotator.rotator import Rotator
    
except ImportError:
    warnings.warn("Failed to import hardware modules")
    
class OSIRISAlias:
    """
    OSIRIS Alias to make image aquisition compatible with FPWFSC API
    """
    def __init__(self):
        self.OSIRIS = OSIRIS()
        self._take_image = self.OSIRIS.take_image


    def take_image(self):
        img_hdu = self._take_image()
        return img_hdu.data

class NIRC2Alias():
    """
    NIRC2 Alias to make image aquisition compatible with FPWFSC API
    """
    def __init__(self):
        self.camera = NIRC2('k2')
        #self.NIRC2.get_parameters()
        self._take_image = self.camera.take_image

    def take_image(self):
        img_hdu = self._take_image()[0]
        print(type(img_hdu))
        return img_hdu.data


class AOSystemAlias:
    def __init__(self):
        """
        Open-loop AO System Interface
        """
        self.AO = xd.XineticsDeformableMirror()
        self.ttm = SHWFSFieldSteeringMirror()
        self._closed = False

    def set_dm_data(self, shape, modify_existing=False):
        return self.AO.set_voltages(shape)

    def get_dm_data(self):
        return self.AO.get_voltages()

class ClosedAOSystemAlias:
    def __init__(self,rotation_angle_dm = 0, flip_x = False, flip_y = False):
        """
        Closed-loop AO System Interface
        """
        self.AO = KeckAO("k2", use_network_qfix=True)
        self.AO.setup()
        self.dm = self.AO.dm # Alpao DM
        self.ttm = self.AO.shwfs 
        self.current_cog_file = self.AO.shwfs.get_current_centroid_origins_filename()
        self.cur_cog = self.AO.shwfs.get_centroid_origins_from_channels(shape_requested='vector')
        
        self.recon = ReconstructorManager(prefix=self.AO.prefix, ao=self.AO)

        # Update to HAKA
        self.dm_shape = self.AO.dm.get_actuators_map().shape
        self.diameter_pupil_act = self.dm_shape[0] 
        self.center_pupil_act = self.diameter_pupil_act // 2 
        self.Nact = self.diameter_pupil_act

        self.rotation_angle_dm = rotation_angle_dm
        self.flip_x = flip_x
        self.flip_y = flip_y
      
        self._closed = True
        self.grid = hcipy.make_uniform_grid([self.Nact, self.Nact], 1, 0)
        
        # load influence functions
        plate_scale = self.AO.shwfs.get_plate_scale()
        if plate_scale == '57x1.50':
            imat = self.AO.rec.default_im_filename_k2_150
        elif plate_scale == '57x0.75':
            imat = self.AO.rec.default_im_filename_k2_075
        else:
            imat = self.AO.rec.default_im_filename_k2_29

        self.infmat_filename = f'{self.recon.get_interaction_matrices_folder_path()}/{imat}'

    def set_dm_data(self, cog_data):
        """Use 
        Parameters
        ----------
        cog_data : array
            array of centroid offsets to apply to the DM. 
        """
        # now we need to write the cog file and load the cog file
        #print(cog_data.shape)
        #ipdb.set_trace()
        saved_filename = self.AO.shwfs.save_centroid_origins_file(cog_data, filename='SAN_Centroids')
        self.AO.shwfs.set_centroid_origins_to_channels(cog_data)
        self.AO.shwfs.load_centroid_origins(saved_filename)
        return

    def get_dm_data(self):
        """
        Returns the current centroid offsets

        Returns
        -------
        ndarray
            current "cog", centroid offsets
        """
        
        cogfile_data = self.AO.shwfs.get_centroid_origins_from_channels(shape_requested="vector")

        return cogfile_data

    def convert_voltage_to_cog(self, phase):
        """
        Parameters
        ----------
        phase: ndarray
            shape to apply to deformable mirror, volts
        """
        # condition shape to be a hcipy Field
        phase = hcipy.Field(phase.ravel(), self.grid)

        # Array with final DM Command (to populate)
        dm_command = np.zeros((self.Nact, self.Nact))

        # the actuators on which we put the pupil
        x_start = int(self.center_pupil_act - self.diameter_pupil_act / 2)
        x_end = int(x_start + self.diameter_pupil_act)

        y_start = int(self.center_pupil_act - self.diameter_pupil_act / 2)
        y_end = int(y_start + self.diameter_pupil_act)

        # filling the array with the actual command
        dm_command[y_start:y_end, x_start:x_end] = phase.shaped

        # dividing by two because we have a reflection and OPD
        dm_volts = dm_command / 2

        binary_map = self.AO.dm.get_binary_actuators_map()
        mask = np.array(binary_map, dtype='bool')
        dm_vec = dm_volts[mask]

        infmat = self.recon.open_interaction_matrix(filename=self.infmat_filename)

        # these are the updated centroid origins
        centroids = np.dot(infmat, dm_vec)

        return centroids
    

class Vampires:
    """
    wrapper for vampires commands that already exist
    """

    def __init__(self):

        self.vcam = shm("vcam1") ##


        # self.filter_name = self.nirc2.get_filters_names()
        # self.wavelength = self.nirc2.get_effective_wavelength()
        # self.pupil_mask_name = self.choose_mask(self.nirc2.get_pupil_mask_name())
        # self.pixel_scale = self.nirc2.get_pixel_scale()
        # self.camera_mode = self.nirc2.get_camera_mode()
        # self.xsize = self.nirc2.get_roi_width()
        # self.ysize = self.nirc2.get_roi_height()

        # Get keywords from the VAMPIRES shm
        shmkwds = self.vcam.get_keywords()
        # Get current filter + dictionnary
        self.filter_name = shmkwds["FILTER01"].strip()
        self.filter_v,self.dict_v = filters.get_filter_info_dict(self.filter_name)
        # Set current wavelength
        self.wavelength = self.dict_v['WAVEAVE']*1e-9
        self.pupil_mask_name = 'subaru'
        self.pixel_scale = 5.9 # need to confirm with Miles
        self.camera_mode = 'standard'
        self.xsize = 536 # check with miles if we can do subwindow
        self.ysize = 536

        pass

    def get_parameters(self, test_time):
        """
        Reads in the current NIRC2 values and sets them to appropriate variables
        """

        # self.filter_name = self.nirc2.get_filters_names()
        # self.wavelength = self.nirc2.get_effective_wavelength()
        # self.pupil_mask_name = self.choose_mask(self.nirc2.get_pupil_mask_name(), test_time)
        # self.pixel_scale = self.nirc2.get_pixel_scale()
        # self.camera_mode = self.nirc2.get_camera_mode()
        # self.xsize = self.nirc2.get_roi_width()
        # self.ysize = self.nirc2.get_roi_height()
                # Get keywords from the VAMPIRES shm
        shmkwds = self.vcam.get_keywords()
        # Get current filter + dictionnary
        self.filter_name = shmkwds["FILTER01"].strip()
        self.filter_v,self.dict_v = filters.get_filter_info_dict(self.filter_name)
        # Set current wavelength
        self.wavelength = self.dict_v['WAVEAVE']*1e-9
        self.pupil_mask_name = 'subaru'
        self.pixel_scale = 5.9 # need to confirm with Miles
        self.camera_mode = 'standard'
        self.xsize = 536 # check with miles if we can do subwindow
        self.ysize = 536

    def choose_mask(self, pmsmask, test_time="Daytime"):
        """
        Reads in the given NIRC2 mask name and converts it to a format readable by F&F code
        """
        mask_name = 'placeholder'

        if test_time == "Daytime":
            if pmsmask == 'open':
                mask_name = 'open'
            elif pmsmask == 'largehex':
                mask_name = 'NIRC2_large_hexagonal_mask'
            elif pmsmask == 'incircle  ':
                mask_name = 'NIRC2_incircle_mask'
            elif pmsmask == 'fixedhex':
                mask_name = 'NIRC2_Lyot_Stop'
            else:
                print('mask name not in known keys')

        elif test_time == "Nighttime":
            if pmsmask == 'open  ':
                mask_name = 'keck'
            elif pmsmask == 'largehex  ':
                mask_name = 'keck+NIRC2_large_hexagonal_mask'
            elif pmsmask == 'incircle  ':
                mask_name = 'keck+NIRC2_incircle_mask'
            elif pmsmask == 'fixedhex  ':
                mask_name = 'keck+NIRC2_Lyot_Stop'
            else:
                print('mask name not in known keys')

        else:
            print('mask name not in known keys')

        return mask_name

    def take_image(self, average = 1):
        """
        Initiate a NIRC2 image with the currently set parameters
        """
        if average == 1 :
            image = self.vcam.get_data(True, True, timeout = 1.).astype(float)
        else:
            im = []
            for i in range(50):
                im.append(self.vcam.get_data(True, True, timeout = 1.).astype(float))
            im = np.array(im)
            image = np.mean(im, axis=0)




        return image



class Palila:
    """
    wrapper for vampires commands that already exist
    """

    def __init__(self):

        self.palilacam = shm("palila") ##aver

        # Fix later
        # self.badpixmap_shm = shm("palila_badpixmap") ##
        # self.badpixmap = self.badpixmap_shm.get_data(True, True, timeout = 1.).astype(float)
        # self.palila_dark_shm = shm("palila_dark") ##
        # self.palila_dark = self.palila_dark_shm.get_data(True, True, timeout = 1.).astype(float)

        self.palila_dark = pf.open('/home/scexao/Documents/FnF/palila_dark.fits')[0].data


        # self.filter_name = self.nirc2.get_filters_names()
        # self.wavelength = self.nirc2.get_effective_wavelength()
        # self.pupil_mask_name = self.choose_mask(self.nirc2.get_pupil_mask_name())
        # self.pixel_scale = self.nirc2.get_pixel_scale()
        # self.camera_mode = self.nirc2.get_camera_mode()
        # self.xsize = self.nirc2.get_roi_width()
        # self.ysize = self.nirc2.get_roi_height()

        # Get keywords from the VAMPIRES shm
        shmkwds = self.palilacam.get_keywords()
        # Get current filter + dictionnary
        self.filter_name = shmkwds["FILTER01"].strip()
        self.palila_filters = {
            # 'OPEN':
            'y-band':1020.0,
            'H-band':1580.0,
            'J-band':1250.0,
            '1550nm, 25nm BW':1550.0,
            '1550nm, 50nm BW':1550.0
        }
        # self.filter_v,self.dict_v = filters.get_filter_info_dict(self.filter_name)
        # Set current wavelength
        self.wavelength = self.palila_filters[self.filter_name]*1e-9
        self.pupil_mask_name = 'subaru'
        self.pixel_scale = 15.3 # need to confirm with Miles
        self.camera_mode = 'standard'
        self.xsize = 320 # check with miles if we can do subwindow
        self.ysize = 256

        pass



    def get_parameters(self, test_time):
        """
        Reads in the current NIRC2 values and sets them to appropriate variables
        """

        # Get keywords from the VAMPIRES shm
        shmkwds = self.palilacam.get_keywords()
        # Get current filter + dictionnary
        self.filter_name = shmkwds["FILTER01"].strip()
        self.palila_filters = {
            # 'OPEN':
            'y-band':1020.0,
            'H-band':1580.0,
            'J-band':1250.0,
            '1550nm, 25nm BW':1550.0,
            '1550nm, 50nm BW':1550.0
        }
        # self.filter_v,self.dict_v = filters.get_filter_info_dict(self.filter_name)
        # Set current wavelength
        self.wavelength = self.palila_filters[self.filter_name]*1e-9
        self.pupil_mask_name = 'subaru'
        self.pixel_scale = 15.3 # need to confirm with Miles
        self.camera_mode = 'standard'
        self.xsize = 320 # check with miles if we can do subwindow
        self.ysize = 256


    def take_image(self, average = 50):
        """
        Initiate a NIRC2 image with the currently set parameters
        """
        if average == 1 :
            image = self.palilacam.get_data(True, True, timeout = 1.).astype(float)
            # image =- self.badpixmap
            image -= self.palila_dark
        else:
            im = []
            for i in range(50):
                im.append(self.palilacam.get_data(True, True, timeout = 1.).astype(float))
            im = np.array(im)
            image = np.mean(im, axis=0)
            # image =- self.badpixmap
            image -= self.palila_dark


        return image






class SCEXAO:

    def __init__(self):
        """
        Basic description of the function.
        """

        # self.shwfs = ShwfsCommands(prefix="k2")
        # self.xinetics = XineticsDeformableMirrorCommands(prefix="k2")

        # self.default_cog = self.shwfs.get_default_centroid_origins_filename()
        # self.current_cog = self.shwfs.get_centroid_origins()



        self.save_cog_name = ""
        self.load_cog_name = ""

        self.dm_command = np.zeros(1)

        self.dm=shm("dm00disp04")

        self.diameter = 44
        self.center = [24,23]
        self.actuator_num = [50,50]
        # self.rotation_angle_dm = 6.25
        self.rotation_angle_dm = 0.


    def make_dm_command(self, phase):
        """Converts the phase estimate to a DM command.

        This function converts the phase estimate to a DM command. This means
        that the phase estimate is rotated to match the DM orientation. Then
        it will be resampled to an array with the appropriate size (actuator_num x actuator_num)
        and put on the active pupil on the DM. It will also take into account
        the reflective nature of the DM and divide the command by 2.

        Parameters
        ----------
        phase : Field
            The phase estimate in volts.
        diameter : integer
            Diameter of active pupil on the DM in actuators.
        center : [integer, integer]
            Position of the center of the active pupil on the dm in actuators [x_pos, y_pos].
        actuator_num : integer
            The number of actuators along one axis of the DM.
        rotation_angle_dm : float
            Rotation angle of the DM in degrees.

        Returns
        ----------
        dm_command : square numpy array
            The DM command derived from the phase estimate.
        '''
        if rotation_angle_dm != 0:

            grid = phase.grid

            shape_phase = phase.shaped.shape

            # rotating the resampled phase
            phase = hcipy.Field(sf.cen_rot(phase.shaped, rotation_angle_dm, np.array(phase.shaped.shape) / 2).ravel(),
            #phase.grid)
        """
        # first we resample the measured phase to the size of the pupil on the actuators
        phase_resampled = sf.fourier_resample_v2(phase, [self.diameter, self.diameter], output_diam=7.9)

        #if rotation_angle_dm > 1e-4:
        if 1 == 1:
            grid = phase_resampled.grid

            # rotating the resampled phase
            phase_resampled = hcipy.Field(sf.cen_rot(phase_resampled.shaped, self.rotation_angle_dm,
                                                     np.array([self.center[1], self.center[0]])).ravel(), grid)
        else:
            pass
            #phase_resampled = phase_resampled.shaped
        # array with the final DM command
        self.dm_command = np.zeros(self.actuator_num)

        # the actuators on which we put the pupil
        x_start = int(self.center[0] - self.diameter / 2)
        x_end = int(x_start + self.diameter)

        y_start = int(self.center[1] - self.diameter / 2)
        y_end = int(y_start + self.diameter)

        # filling the array with the actual command
        self.dm_command[y_start:y_end, x_start:x_end] = phase_resampled.shaped

        # dividing by two because we have a reflection and OPD
        self.dm_command /= 2

        return self.dm_command

    def set_dm_data(self, dmvolts):

        self.dm.set_data(dmvolts.astype(np.float32))
        time.sleep(0.01)

        # -----------------------------------------------------------------------------
        # Close shared memory
        # -----------------------------------------------------------------------------

        # self.dm.close()

        pass

    def close_dm_stream(self):

        self.dm.close()


# class NIRC2:
#    """
#    wrapper for nirc2 commands that already exist
#    """

#    def __init__(self):

#        self.nirc2 = Nirc2LibraryCommands()

#        self.filter_name = self.nirc2.get_filters_names()
#        self.wavelength = self.nirc2.get_effective_wavelength()
#        self.pupil_mask_name = self.choose_mask(self.nirc2.get_pupil_mask_name())
#        self.pixel_scale = self.nirc2.get_pixel_scale()
#        self.camera_mode = self.nirc2.get_camera_mode()
#        self.xsize = self.nirc2.get_roi_width()
#        self.ysize = self.nirc2.get_roi_height()

#        pass

#    def get_parameters(self, test_time):
#        """
#        Reads in the current NIRC2 values and sets them to appropriate variables
#        """

#        self.filter_name = self.nirc2.get_filters_names()
#        self.wavelength = self.nirc2.get_effective_wavelength()
#        self.pupil_mask_name = self.choose_mask(self.nirc2.get_pupil_mask_name(), test_time)
#        self.pixel_scale = self.nirc2.get_pixel_scale()
#        self.camera_mode = self.nirc2.get_camera_mode()
#        self.xsize = self.nirc2.get_roi_width()
#        self.ysize = self.nirc2.get_roi_height()

#    def choose_mask(self, pmsmask, test_time="Daytime"):
#        """
#        Reads in the given NIRC2 mask name and converts it to a format readable by F&F code
#        """
#        mask_name = 'placeholder'

#        if test_time == "Daytime":
#            if pmsmask == 'open':
#                mask_name = 'open'
#            elif pmsmask == 'largehex':
#                mask_name = 'NIRC2_large_hexagonal_mask'
#            elif pmsmask == 'incircle  ':
#                mask_name = 'NIRC2_incircle_mask'
#            elif pmsmask == 'fixedhex':
#                mask_name = 'NIRC2_Lyot_Stop'
#            else:
#                print('mask name not in known keys')

#        elif test_time == "Nighttime":
#            if pmsmask == 'open  ':
#                mask_name = 'keck'
#            elif pmsmask == 'largehex  ':
#                mask_name = 'keck+NIRC2_large_hexagonal_mask'
#            elif pmsmask == 'incircle  ':
#                mask_name = 'keck+NIRC2_incircle_mask'
#            elif pmsmask == 'fixedhex  ':
#                mask_name = 'keck+NIRC2_Lyot_Stop'
#            else:
#                print('mask name not in known keys')

#        else:
#            print('mask name not in known keys')

#        return mask_name

#    def take_image(self):
#        """
#        Initiate a NIRC2 image with the currently set parameters
#        """

#        image = self.nirc2.take_image()

#        return image
# class KeckAO:

#     def __init__(self):
#         """
#         Basic description of the function.
#         """

#         self.shwfs = SHWFS(prefix="k2")
#         self.xinetics = xd.XineticsDeformableMirror(prefix="k2")
#         self.rotator = Rotator(prefix="k2")

#         self.default_cog = self.shwfs.get_default_centroid_origins_filename()
#         self.current_cog = self.shwfs.get_centroid_origins()

#         self.save_cog_name = ""
#         self.load_cog_name = ""
#         self.cog_name = ""
#         self.ciog_data = ""

#         self.dm_command = np.zeros(1)

#     def get_rotator_mode(self):
#         """"
#         This function gets the mode of the rotator. Note: F&F currently only runs in vertical angle mode.
#         """
#         rotator_mode = self.rotator.get_mode()
#         return rotator_mode

#     def get_rotator_angle(self):
#         """
#         Read and update the pupil angle of the rotator. This is needed as an input for the F&F algorithm model.
#         """
#         pupil_angle = self.rotator.get_pupil_angle()

#         return pupil_angle

#     def revert_cog(self):
#         """
#         Revert the current cog file to the start of the night
#         """
#         # get the default centroid origins to reload
#         self.default_cog = self.shwfs.get_default_centroid_origins_filename()
#         self.shwfs.load_centroid_origins(self.default_cog)

#     def get_cog_filename(self):
#         """
#         Get the current centroid origin filename
#         """
#         self.cog_name = self.shwfs.get_current_centroid_origins_filename()
#         return self.cog_name

#     def open_cog(self, cog_name, shape_requested = "vector"):
#         """
#         Open and read the give cog file, return an array
#         """
#         self.cog_data = self.shwfs.open_centroid_origins_file(cog_name, shape_requested=shape_requested)
#         return self.cog_data

#     def load_cog(self, load_cog_name):
#         """
#         Load a cog file given an input filename
#         """
#         self.load_cog_name = load_cog_name
#         self.shwfs.load_centroid_origins(self.load_cog_name)

#         return self.load_cog_name

#     def save_cog(self, save_cog_name, cog = None, timestamp = True):
#         """
#         Save the current cog file with a generated filename
#         """
#         if cog is not None:
#             self.current_cog = cog
#         else:
#             self.current_cog = self.shwfs.get_centroid_origins()


#         save_filename = save_cog_name

#         saved_filename = self.shwfs.save_centroid_origins_file(self.current_cog, filename=save_filename, add_timestamp=timestamp)

#         return saved_filename

#     def open_influence_matrix(self):
#         return self.shwfs.open_influence_matrix('24.imx')


#     def get_dm_actuator_map(self):
#         """
#         Pulls the binary DM actuator map and returns as an array
#         """
#         dm_actuator_map = self.xinetics.get_binary_actuators_map()

#     def make_dm_command(self, phase, diameter, center, actuator_num, rotation_angle_dm = 0 , flip_x = False, flip_y = False):
#         """Converts the phase estimate to a DM command.

#     This function converts the phase estimate to a DM command. This means
#     that the phase estimate is rotated to match the DM orientation. Then
#     it will be resampled to an array with the appropriate size (actuator_num x actuator_num)
#     and put on the active pupil on the DM. It will also take into account
#     the reflective nature of the DM and divide the command by 2.

#     Parameters
#     ----------
#     phase : Field
#         The phase estimate in volts.
#     diameter : integer
#         Diameter of active pupil on the DM in actuators.
#     center : [integer, integer]
#         Position of the center of the active pupil on the dm in actuators [x_pos, y_pos].
#     actuator_num : integer
#         The number of actuators along one axis of the DM.
#     rotation_angle_dm : float
#         Rotation angle of the DM in degrees.

#     Returns
#     ----------
#     dm_command : square numpy array
#         The DM command derived from the phase estimate.
#     '''
#     if rotation_angle_dm != 0:

#         grid = phase.grid

#         shape_phase = phase.shaped.shape

#         # rotating the resampled phase
#         phase = hcipy.Field(sf.cen_rot(phase.shaped, rotation_angle_dm, np.array(phase.shaped.shape) / 2).ravel(),
#         #phase.grid)
#     """
#     # first we resample the measured phase to the size of the pupil on the actuators
#         phase_resampled = sf.fourier_resample(phase, [diameter, diameter])

#         if rotation_angle_dm != 0:
#             grid = phase_resampled.grid

#             # rotating the resampled phase
#             phase_resampled = hcipy.Field(sf.cen_rot(phase_resampled.shaped, rotation_angle_dm,
#                                                      np.array([center[1], center[0]])).ravel(), grid)

#         if flip_x == True:
#             grid = phase_resampled.grid
#             phase_resampled = hcipy.Field(np.flip(phase_resampled.shaped, axis = 0).ravel(), grid)
#         if flip_y == True:
#             grid = phase_resampled.grid
#             phase_resampled = hcipy.Field(np.flip(phase_resampled.shaped, axis = 1).ravel(), grid)

#         # array with the final DM command
#         self.dm_command = np.zeros((actuator_num, actuator_num))

#         # the actuators on which we put the pupil
#         x_start = int(center[0] - diameter / 2)
#         x_end = int(x_start + diameter)

#         y_start = int(center[1] - diameter / 2)
#         y_end = int(y_start + diameter)

#         # filling the array with the actual command
#         self.dm_command[y_start:y_end, x_start:x_end] = phase_resampled.shaped

#         # testing if flipping the axis of the DM improves the result.
#         # all three options (y-, x-axis, both) were tried and did not improve the loop
#         # dm_command = dm_command[:,::-1]

#         # dividing by two because we have a reflection and OPD
#         self.dm_command /= 2



#         return self.dm_command


# class KeckAO:

#    def __init__(self):
#        """
#        Basic description of the function.
#        """

#        self.shwfs = SHWFS(prefix="k2")
#        self.xinetics = xd.XineticsDeformableMirror(prefix="k2")

#        self.default_cog = self.shwfs.get_default_centroid_origins_filename()
#        self.current_cog = self.shwfs.get_centroid_origins()

#        self.save_cog_name = ""
#        self.load_cog_name = ""

#        self.dm_command = np.zeros(1)

#    def revert_cog(self):
#        """
#        Revert the current cog file to the start of the night
#        """
#        # get the default centroid origins to reload
#        self.default_cog = self.shwfs.get_default_centroid_origins_filename()
#        self.shwfs.load_centroid_origins(self.default_cog)

#    def get_cog_filename(self):
#        """
#        Get the current centroid origin filename
#        """
#        self.cog_name = self.shwfs.get_current_centroid_origins_filename()
#        return self.cog_name

#    def open_cog(self, cog_name, shape_requested = "vector"):
#        """
#        Open and read the give cog file, return an array
#        """
#        self.cog_data = self.shwfs.open_centroid_origins_file(cog_name, shape_requested=shape_requested)
#        return self.cog_data

#    def load_cog(self, load_cog_name=""):
#        """
#        Load a cog file given an input filename
#        """
#        if load_cog_name == "" :
#            load_cog_name = self.shwfs.get_current_centroid_origins_filename()


#        self.load_cog_name = load_cog_name
#        self.shwfs.load_centroid_origins(self.load_cog_name)

#        return self.load_cog_name


#    def save_cog(self, save_cog_name="", cog=""):
#        """
#        Save the current cog file with a generated filename
#        """
#        if cog == "":
#            self.current_cog = self.shwfs.get_centroid_origins()
#        else:
#            self.current_cog = cog

#        save_filename = save_cog_name

#        # if no filename is given, a filename with timestamp will be generated
#        if save_filename == "":
#            save_filename = self.shwfs.save_centroid_origins_file(self.current_cog)
#        else:
#            self.shwfs.save_centroid_origins_file(self.current_cog, filename=save_filename, add_timestamp=False)

#        return save_filename

#    def open_influence_matrix(self):
#        self.shwfs.open_influence_matrix('24.imx')
#    def get_dm_actuator_map(self):
#        """
#        Pulls the binary DM actuator map and returns as an array
#        """
#        dm_actuator_map = self.xinetics.get_binary_actuators_map()

#    def make_dm_command(self, phase, diameter, center, actuator_num, rotation_angle_dm = 0, flip_x = False, flip_y = False):
#        """Converts the phase estimate to a DM command.

#    This function converts the phase estimate to a DM command. This means
#    that the phase estimate is rotated to match the DM orientation. Then
#    it will be resampled to an array with the appropriate size (actuator_num x actuator_num)
#    and put on the active pupil on the DM. It will also take into account
#    the reflective nature of the DM and divide the command by 2.

#    Parameters
#    ----------
#    phase : Field
#        The phase estimate in volts.
#    diameter : integer
#        Diameter of active pupil on the DM in actuators.
#    center : [integer, integer]
#        Position of the center of the active pupil on the dm in actuators [x_pos, y_pos].
#    actuator_num : integer
#        The number of actuators along one axis of the DM.
#    rotation_angle_dm : float
#        Rotation angle of the DM in degrees.

#    Returns
#    ----------
#    dm_command : square numpy array
#        The DM command derived from the phase estimate.
#    '''
#    if rotation_angle_dm != 0:

#        grid = phase.grid

#        shape_phase = phase.shaped.shape

#        # rotating the resampled phase
#        phase = hcipy.Field(sf.cen_rot(phase.shaped, rotation_angle_dm, np.array(phase.shaped.shape) / 2).ravel(),
#        #phase.grid)
#    """
#    # first we resample the measured phase to the size of the pupil on the actuators
#        phase_resampled = sf.fourier_resample(phase, [diameter, diameter])

#        if rotation_angle_dm != 0:
#            grid = phase_resampled.grid

#            # rotating the resampled phase
#            phase_resampled = hcipy.Field(sf.cen_rot(phase_resampled.shaped, rotation_angle_dm,
#                                                     np.array([center[1], center[0]])).ravel(), grid)
#        if flip_x == True:
#            grid = phase_resampled.grid
#            phase_resampled = hcipy.Field(np.flip(phase_resampled.shaped, axis = 0).ravel(), grid)
#        if flip_y == True:
#            grid = phase_resampled.grid
#            phase_resampled = hcipy.Field(np.flip(phase_resampled.shaped, axis = 1).ravel(), grid)

#        # array with the final DM command
#        self.dm_command = np.zeros(actuator_num)

#        # the actuators on which we put the pupil
#        x_start = int(center[0] - diameter / 2)
#        x_end = int(x_start + diameter)

#        y_start = int(center[1] - diameter / 2)
#        y_end = int(y_start + diameter)

#        # filling the array with the actual command
#        self.dm_command[y_start:y_end, x_start:x_end] = phase_resampled.shaped

#        # testing if flipping the axis of the DM improves the result.
#        # all three options (y-, x-axis, both) were tried and did not improve the loop
#        # dm_command = dm_command[:,::-1]

#        # dividing by two because we have a reflection and OPD
#        self.dm_command /= 2

#        return self.dm_command



