from hcipy import *
import matplotlib.pyplot as plt
import numpy as np
pupil_grid = make_pupil_grid(128)
telescope_pupil_gen = make_gmt_aperture(normalized = True)
pupil = telescope_pupil_gen(pupil_grid)

imshow_field(pupil)
#plt.imshow(pupil.shaped, origin = 'lower')
plt.show()
