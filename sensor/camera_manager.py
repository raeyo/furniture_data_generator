from pyrep.objects.dummy import Dummy
from pyrep.backend import sim
from sensor.mycamera import MyCamera
from pyrep.const import RenderMode

import numpy as np
from enum import Enum
class CameraInfo():
    def __init__(self, resolution, perspective_angle, angle_range):
        self.resolution = resolution
        self.perspective_angle = perspective_angle
        self.angle_range = angle_range

class CameraType(Enum):
    # Azure = CameraInfo([2048, 1536], 90, [87, 93]) 4:3
    
    Azure = CameraInfo([1280, 720], 90, [87, 93]) # 16:9
    Zivid_ML = CameraInfo([1920, 1200], 33, [30, 42])
    Zivid_M = CameraInfo([1920, 1200], 33, [30, 36])

class CameraManager(object):
    """
    move cameras
    get images
    manage targets
    """
    def __init__(self, camera_type: CameraInfo, pr):
        self.controller = Dummy("camera_control")
        self.rot_base = Dummy("camera_rot_base")

        self.pr = pr        
        
        self._resolution = camera_type.value.resolution
        self._perspective_angle = camera_type.value.perspective_angle
        self._pangle_range = camera_type.value.angle_range

        # changing by labeling type
        self.main_camera = MyCamera("main", self._resolution)
        self.seg_camera = MyCamera("seg_mask", self._resolution)
        self.hole_camera = MyCamera("hole_mask", self._resolution)
        self.asm_camera = MyCamera("asm_mask", self._resolution)
        # check before data generate(in scene)
        self._initial_pose = self.controller.get_pose(relative_to=self.rot_base)

    def reset(self):
        # randomize rotation base
        self.controller.set_pose(self._initial_pose, relative_to=self.rot_base)
        
    def _set_perspective_angle(self, angle):
        self.main_camera.set_perspective_angle(angle)
        self.seg_camera.set_perspective_angle(angle)
        self.hole_camera.set_perspective_angle(angle)
        self.asm_camera.set_perspective_angle(angle)
        self._perspective_angle = angle

    def set_random_angle(self):
        rand_pangle = np.random.uniform(self._pangle_range[0], self._pangle_range[1])


    def get_perspective_angle(self):
        return self._perspective_angle

    def get_resolution(self):
        return self.main_camera.get_resolution()

    def get_distance(self):
        base_pos = np.array(self.rot_base.get_position())
        cam_pos = np.array(self.controller.get_position())
        distance = np.linalg.norm(base_pos - cam_pos)
        return distance
    
    def set_rotation_base_to(self, obj):
        self.rot_base.set_pose(obj.get_pose())
        noise_pos = list(np.random.rand(3)/10)
        self.rot_base.set_position(noise_pos, relative_to=self.rot_base)


    def set_position(self, position, relative_to=None):
        self.controller.set_position(position, relative_to=relative_to)

    def set_activate(self, is_activate):
        self.main_camera.set_activate(is_activate)
        self.seg_camera.set_activate(is_activate)
        self.hole_camera.set_activate(is_activate)
        self.asm_camera.set_activate(is_activate)

    # changing by labeling
    def capture(self):
        self.main_rgb, self.main_depth = self.main_camera.get_image()
        self.seg_rgb, _ = self.seg_camera.get_image("rgb")
        hole_rgb, _ = self.hole_camera.get_image("rgb")
        self.hole_mask = hole_rgb[:, :, 0] > 0.5
        _, asm_depth = self.asm_camera.get_image("depth")
        self.asm_mask = asm_depth < 0.98
        
    def get_images(self):
        return self.main_rgb, self.main_depth, self.seg_rgb, self.hole_mask, self.asm_mask

    # povray {focalBlur {false} focalDist {2.00} aperture{0.05} blurSamples{10}}