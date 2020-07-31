from pyrep.objects.dummy import Dummy
from pyrep.backend import sim
from sensor.mycamera import MyCamera
import numpy as np
from pyrep.const import RenderMode
class CameraManager(object):
    """
    move cameras
    get images
    manage targets
    """
    def __init__(self, resolution, perspective_angle, pr):
        self.controller = Dummy("camera_control")
        self.rot_base = Dummy("camera_rot_base")

        self.pr = pr        
        
        self._resolution = resolution
        self._perspective_angle = perspective_angle

        #TODO: changing by labeling type
        self.main_camera = MyCamera("main", resolution)
        self.seg_camera = MyCamera("seg_mask", resolution)
        self.hole_camera = MyCamera("hole_mask", resolution)
        self.asm_camera = MyCamera("asm_mask", resolution)
        #TODO: check before data generate(in scene)
        self._initial_pose = self.controller.get_pose(relative_to=self.rot_base)

    def reset(self):
        #TODO: randomize rotation base
        self.controller.set_pose(self._initial_pose, relative_to=self.rot_base)
        self.main_camera.set_render_mode(RenderMode.OPENGL3)
        
    def set_perspective_angle(self, angle):
        self.main_camera.set_perspective_angle(angle)
        self.seg_camera.set_perspective_angle(angle)
        self.hole_camera.set_perspective_angle(angle)
        self.asm_camera.set_perspective_angle(angle)
        self._perspective_angle = angle

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

    def set_position(self, position, relative_to=None):
        self.controller.set_position(position, relative_to=relative_to)

    #TODO: changing by labeling
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