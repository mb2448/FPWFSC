# allow control of the k2 ao bench
from kaotools.epics import k2bench
from kaotools.epics.k2bench import EPICS_PREFIX
k2bench.enable_control()
import numpy as np
import hcipy
import guis.fast_and_furious.support_functions as sf
import matplotlib.pyplot as plt
import guis.fast_and_furious.make_keck_aperture as mka

import guis.fast_and_furious.classes as ff_c
import guis.fast_and_furious.hardware as hardware

hw = hardware.KeckAO()

from aosys.shwfs import shwfs

shwfs = shwfs.SHWFS(prefix=EPICS_PREFIX)

nirc2 = hardware.NIRC2()


def run_fastandfurious_test():
    FF_ini = 'FF_software.ini'
    FF_spec = 'FF_software.spec'
    settings = sf.validate_configfile(FF_ini, FF_spec)

    # ----------------------------------------------------------------------
    # Control Loop parameters
    # ----------------------------------------------------------------------
    Niter = settings['LOOP_SETTINGS']['N iter']
    gain = settings['LOOP_SETTINGS']['gain']
    leak_factor = settings['LOOP_SETTINGS']['leak factor']
    chosen_mode_basis = settings['LOOP_SETTINGS']['Used mode basis']
    Nmodes = settings['LOOP_SETTINGS']['Number of modes']

    # ----------------------------------------------------------------------
    # Optical model parameters
    # ----------------------------------------------------------------------
    chosen_aperture = settings['MODELLING']['aperture']
    chosen_aperture = nirc2.pupil_mask_name
    wavelength = settings['MODELLING']['wavelength (m)']
    rotation_angle_aperture = settings['MODELLING']['rotation angle aperture (deg)']
    Npix_pup = settings['MODELLING']['N pix pupil']
    Npix_foc = settings['MODELLING']['N pix focal']
    grid_diameter = settings['MODELLING']['grid diameter (m)']
    mas_pix = settings['MODELLING']['pixel scale (mas/pix)']
    epsilon = settings['FF_SETTINGS']['epsilon']
    flux = 1e5
    diameter_pupil_act = 21
    center_pupil_act = [10,10]
    Nact = 21
    rotation_angle_dm = settings['MODELLING']['rotation angle dm (deg)']
    xcen = 248
    ycen = 270
    # ----------------------------------------------------------------------
    # Setting up FnF with the Optical Model
    # --------------------------------------------------------------
    # --------



        #flux =  # estimate the flux for the optical model from the initial nirc2 image

    Aperture = ff_c.Aperture(Npix_pup=Npix_pup, diameter=grid_diameter,
                             aperturename=chosen_aperture,
                             rotation_angle_aperture=rotation_angle_aperture,
                             aperture_generator=mka.get_aperture)

    OpticalModel = ff_c.SystemModel(aperture=Aperture,
                                    Npix_foc=Npix_foc,
                                    mas_pix=mas_pix,
                                    wavelength=wavelength,
                                    flux=flux)

    FnF = ff_c.FastandFurious(SystemModel=OpticalModel,
                              leak_factor=leak_factor,
                              gain=gain,
                              epsilon=epsilon,
                              chosen_mode_basis=chosen_mode_basis,
                              number_of_modes=Nmodes)

    data_raw = nirc2.take_image()[0].data
    data_ref=sf.reduce_images(data_raw, xcen=xcen, ycen=ycen, npix=Npix_foc, refpsf=OpticalModel.ref_psf.shaped)
    print(Npix_foc)
    A = OpticalModel.ref_psf.shaped
    print(A.shape[0])

    # ----------------------------------------------------------------------
    # Setting up the detector
    # ----------------------------------------------------------------------
    # Detector = hw.NIRC2()

    # ----------------------------------------------------------------------
    # Initialize
    # ----------------------------------------------------------------------
    phase_error = np.zeros(Npix_pup ** 2)
    phase_DM = phase_error
    # Create the wavefront
    OpticalModel.update_pupil_wavefront(phase_error)

    # generating the first reference image

    # Take first image
    FnF.initialize_first_image(data_ref)

    # ----------------------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------------------

    RMS_measurements = np.zeros(Niter)
    RMS_measurements[RMS_measurements == 0] = np.nan

    SRA_measurements = np.zeros(Niter)
    SRA_measurements[SRA_measurements == 0] = np.nan

    VAR_measurements = np.zeros(Niter)
    VAR_measurements[VAR_measurements == 0] = np.nan

    #----------------------------------------------------------------------
    # looping through mode basis to check if theory and bench match
    # ----------------------------------------------------------------------

    current_cog_file = shwfs.get_current_centroid_origins_filename()
    cur_cog = shwfs.open_centroid_origins_file(current_cog_file, shape_requested='vector')

    mode_basis = hcipy.make_zernike_basis(5, grid_diameter, Aperture.pupil_grid, 5)
    # orthogonalizing the mode basis for the specific aperture
    mode_basis = sf.orthonormalize_mode_basis(mode_basis, Aperture.aperture)
    amplitude = 0.5
    for mode, i in zip(mode_basis, np.arange(len(mode_basis))):


        # creating the phase that will be introduced
        phase_rad = mode * amplitude
        volts = phase_rad * FnF.wavelength / 10 ** -9 / 600
        dm_volts = hw.make_dm_command(volts, diameter_pupil_act, center_pupil_act, Nact, rotation_angle_dm)
        # need to convert phase_DM to 349 actuators
        actmap = np.array(shwfs.dm.get_binary_actuators_map(), dtype='bool')

        dm_vec = dm_volts[actmap]
        infmat = shwfs.open_influence_matrix('24.imx')
        cents = np.dot(infmat, dm_vec)  # these are the updated centroid origins

        # get the current centoid origins

        new_cents = cur_cog + cents

        # now we need to write the cog file and load the cog file
        fn = shwfs.save_centroid_origins_file(new_cents, filename='FandF')
        shwfs.load_centroid_origins(fn)
        image = nirc2.take_image()[0].data
        pupil_wf = hcipy.Wavefront(Aperture.aperture * np.exp(1j * phase_rad),
                             wavelength=FnF.wavelength)
        focal_wf = OpticalModel.propagator(pupil_wf)

        # getting the images by theory and practice
        image_theory = focal_wf.power

        image_bench = sf.reduce_images(image, xcen=xcen, ycen=ycen, npix=Npix_foc, refpsf=OpticalModel.ref_psf.shaped)

        plt.figure(figsize=(8, 8))

        plt.subplot(2, 2, 1)
        hcipy.imshow_field(np.log10(image_theory / image_theory.max()), vmin=-3)
        plt.colorbar()
        plt.title('theory')

        plt.subplot(2, 2, 2)
        plt.imshow(np.log10(np.abs(image_bench) / image_bench.max()), vmin=-3)
        plt.colorbar()
        plt.title('bench')

        max_theory = np.max(np.abs(phase_rad))

        plt.subplot(2, 2, 3)
        hcipy.imshow_field(phase_rad, cmap='bwr', vmin=-max_theory, vmax=max_theory)
        plt.colorbar()

        plt.title('theory')

        max_bench = np.max(np.abs(dm_volts))

        plt.subplot(2, 2, 4)
        plt.imshow(dm_volts, origin='lower', cmap='bwr',
                   vmin=-max_bench, vmax=max_bench)

        plt.colorbar()

        plt.title('Applied command')

        plt.show()
        # converting the volt
    hw.revert_cog()


if __name__ == "__main__":
    run_fastandfurious_test()