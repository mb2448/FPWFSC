from aoscripts.ao_systems.k2ao import K2AO


class NIRC2Alias():
    """
    NIRC2 Alias to make image aquisition compatible with FPWFSC API
    """
    def __init__(self):
        aosys = K2AO()
        aosys.setup()
        self.camera = aosys.nirc2
        #self.NIRC2.get_parameters()
        self._take_image = self.camera.take_image

    def take_image(self):
        img_hdu = self._take_image()
        return img_hdu[0].data

class K2AOAlias():
    def __init__(self):
        self.aosys = K2AO()
        self.aosys.setup()
    
    def offset_tiptilt(self, x, y):
        offsets = self.aosys.dtt.get_offsets()
        xoff = offsets['Offset X']
        yoff = offsets['Offset Y']

        new_x = xoff + x
        new_y = yoff + y
        self.aosys.dtt.set_offsets(new_x, new_y)
        return

    def zero_tiptilt(self):
        """Zero out the tip/tilt"""
        self.aosys.dtt.clear_close_loop_offsets()
        return

if __name__ == "__main__":
    camera = NIRC2Alias()
    AOsystem = K2AOAlias()
