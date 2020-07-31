from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.backend import sim
import numpy as np
from scipy.spatial.transform import Rotation as R

def degree2radian(degree):
    return (degree/180)*np.pi

class MyCamera(object):
    """
    Mycamera(vision sensor)
    """
    def __init__(self, name, resolution, perspective_angle=90):
        self.camera = VisionSensor(name) 

        # camera intrinsic
        self._set_resolution(resolution)
        self.set_perspective_angle(perspective_angle)
    
    def _set_resolution(self, resolution):
        self.camera.set_resolution(resolution)
        self._resolution = resolution
    def get_resolution(self):
        return self._resolution

    def get_pose(self, relative_to=None):
        """
        return pose[x, y, z, qx, qy, qz, w]
        """
        try:
            if relative_to == None:
                return self.camera.get_pose()
            else:
                return self.camera.get_pose(relative_to=relative_to)

        except :
            return self.camera.get_pose(relative_to=relative_to)
    def get_position(self, relative_to=None):
        """
        return position[x, y, z]
        """
        try:
            if relative_to == None:
                return self.camera.get_position()
            else:
                return self.camera.get_position(relative_to=relative_to)

        except :
            return self.camera.get_position(relative_to=relative_to)

    def set_position(self, position, relative_to=None):
        try:
            if relative_to == None:
                self.camera.set_position(position)
            else:
                self.camera.set_position(position, relative_to=relative_to)
        except:
            self.camera.set_position(position, relative_to=relative_to)

    def set_perspective_angle(self, angle):
        self.camera.set_perspective_angle(angle)
        self._perspective_angle = angle
    def get_perspective_angle(self):
        return self._perspective_angle

    def set_render_mode(self, render_mode):
        self.camera.set_render_mode(render_mode)

    def get_image(self, img_type="all"):
        """get images from vision sensor in scene
        depth image is 0 ~ 1 and 0.998... is not detected value

        Returns:
        """
        if img_type == "all":
            rgb_image = self.camera.capture_rgb()
            depth_image = self.camera.capture_depth()
        elif img_type == "rgb":
            rgb_image = self.camera.capture_rgb()
            depth_image = None
        elif img_type == "depth":
            rgb_image = None
            depth_image = self.camera.capture_depth()

        return rgb_image, depth_image
