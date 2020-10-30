from robot.mypanda import MyPanda
from pyrep.objects.shape import Shape
from pyrep.const import TextureMappingMode
import numpy as np
import random

from const import *
from externApi.fileApi import *
from externApi.imageProcessApi import *

import cv2

class RobotManager(object):
    def __init__(self, gripper_robot_num=1, no_gripper_num=2):
        self.robots = [MyPanda(i) for i in range(gripper_robot_num)]
        self.robots += [MyPanda(i, is_gripper=False) for i in range(no_gripper_num)]
        self.visible = []
        for robot in self.robots:
            self.visible += robot.get_visible_objects()
        for i, robot in enumerate(self.robots[gripper_robot_num:]):
            hand_name = "panda_hand_visual#" + str(i)
            finger_name = "panda_finger_visual#" + str(i)
            visible_grasp = [Shape(hand_name), Shape(finger_name)]
            self.visible += visible_grasp
        self.respondable = []
        for robot in self.robots:
            self.respondable += robot.get_respondable_objects()

    def set_seg_objects(self, seg_objects):
        self.seg_objects = seg_objects
        
    def set_seg_state(self, is_seg):
        for seg in self.seg_objects:
            seg.set_renderable(is_seg)
        for vis in self.visible:
            vis.set_renderable(not is_seg)

    def randomize_pose(self):
        for robot in self.robots:
            robot.set_random_pose()

    def randomize_ee_pose(self, pr):
        for robot in self.robots:
            random_position = robot.get_random_ee_pose()
            random_orientation = np.random.rand(3) * np.pi
            robot.move_to_target_pose(pr, target_position=random_position,
                                      )#target_orientation=random_orientation)

    def get_random_robot(self):
        return random.choice(self.robots)

    def reset(self):
        for robot in self.robots:
            robot.reset()

class TextureManager(object):
    def __init__(self, process_id, pr):
        self.process_id = process_id
        self.pr = pr
        self._color_variance = 0.05
        self.texture_shapes = []
        self.textures = self._load_texture_files()
        
    def _get_random_color(self, refer=None):
        """[get random color]
        
        Keyword Arguments:
            refer {[list(3)(0-1)]} -- [reference color] (default: {None})
        """
        if not type(refer) == list: # no reference color
            return list(np.random.rand(3))
        else:
            reference_color = np.array(refer)
            color_min = np.clip(reference_color - self._color_variance, 0,1) 
            color_max = np.clip(reference_color + self._color_variance, 0,1)
            random_color = list(np.random.uniform(color_min, color_max))
            return random_color        
    
    def _load_texture_files(self):
        textures = {}
        for p in TextureType:
            textures[p.name] = get_file_list(p.value)

        return textures

    def _get_random_texture(self, texture_type):
        texture_path = random.choice(self.textures[texture_type.name])

        return texture_path

    def _create_texture(self, texture_path):
        try:
            shape, texture = self.pr.create_texture(filename=texture_path)
        except:
            return False
        shape.set_detectable(False)
        shape.set_renderable(False)
        self.texture_shapes.append(shape)

        return texture
    
    def _get_mixed_randomized_teuxture(self, texture_type):
        num_h = random.randint(1, 3)
        num_w = random.randint(1, 3)
        mix_texture = []
        for h in range(num_h):
            mix_row = [list(cv2.imread(self._get_random_texture(texture_type))) for w in range(num_w)]
            mix_row = np.hstack(mix_row)
            mix_texture.append(mix_row)
        mix_texture = np.vstack(mix_texture)
        save_path = os.path.join(TextureType.mixed_texture.value, 'mix_texture_' + str(self.process_id) + '.png')
        cv2.imwrite(save_path, mix_texture)
        return save_path

    def _get_grad_randomized_texture(self, texture_path, refer):
        """create randomize texture in scene
        1. get random gray texture from texture file list
        2. get two random color(can be refer)
        3. create gradation texture image
        4. create texture shape(in the scene) 
        
        Keyword Arguments:
            refer {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        """
        #region create gradient randomize texture
        first_color = self._get_random_color(refer=refer)
        if refer==None:
            refer = list(1 - np.array(first_color)) # complement color
        second_color = self._get_random_color(refer=refer)
        rand_texture_path = create_randomize_image(texture_path,
                                              first_color,
                                              second_color,
                                              self.process_id)        
        while not type(rand_texture_path) == str:
            print("[ERROR] {} texture occur error!".format(texture_path))
            texture_path = self._get_random_gray_texture()
            first_color = self._get_random_color(refer=refer)
            second_color = self._get_random_color(refer=refer)
            rand_texture_path = create_randomize_image(texture_path,
                                                first_color,
                                                second_color,
                                                self.process_id)        
        #endregion
             
        return rand_texture_path
    
    def get_randomize_texture(self, texture_type, refer=None):
        if texture_type == TextureType.gray_texture:
            rand_texture_path = self._get_random_texture(texture_type)
            rand_texture_path = self._get_grad_randomized_texture(rand_texture_path, refer=refer)
        elif texture_type == TextureType.mixed_texture:
            rand_texture_path = self._get_mixed_randomized_teuxture(TextureType.crawled_texture)
        else:
            rand_texture_path = self._get_random_texture(texture_type)

        texture = False
        while type(texture) == bool:
            texture = self._create_texture(rand_texture_path)

        return texture
    
    def _set_random_texture(self, obj_visible, texture):
        mapping_ind = np.random.randint(4)
        uv_scale = list(np.random.uniform((0, 0), (5, 5)))
        if not type(obj_visible) == list:    
            obj_visible.set_texture(texture=texture,
                                    mapping_mode=TextureMappingMode(mapping_ind),
                                    uv_scaling=uv_scale,
                                    repeat_along_u=True,
                                    repeat_along_v=True)  
        else:
            for vis in obj_visible:
                vis.set_texture(texture=texture,
                                mapping_mode=TextureMappingMode(mapping_ind),
                                uv_scaling=uv_scale,
                                repeat_along_u=True,
                                repeat_along_v=True)  

    def set_random_texture(self, obj_visible, texture_type: TextureType, refer=None):
        texture = self.get_randomize_texture(texture_type, refer=refer)
        self._set_random_texture(obj_visible, texture)
    
    def reset(self):
        for sh in self.texture_shapes:
            sh.remove()
        self.texture_shapes = []

class LabelingManager(object):
    def __init__(self, class_num, RGB=-1):
        self.class_num = class_num
        self.color_range = 255 / self.class_num
        self.class_colors = []
        for class_id in range(self.class_num): # 5
            rgb_min = [0, 0, 0]
            rgb_max = [0, 0, 0]
            if RGB == -1:
                rgb_min = [self.color_range * class_id + 1, self.color_range * class_id + 1 , self.color_range * class_id + 1] 
                rgb_max = [self.color_range * (class_id + 1), self.color_range * (class_id + 1), self.color_range * (class_id + 1)]
            else:
                rgb_min[RGB] = self.color_range * class_id + 1
                rgb_max[RGB] = self.color_range * (class_id + 1)

            self.class_colors.append((rgb_min, rgb_max))
        
        self.used_colors = []

    def get_class_color(self, class_id):
        color_range = self.class_colors[class_id]
        rgb_low = color_range[0]
        rgb_max = color_range[1]
        seg_color = np.uint8(np.random.uniform(rgb_low, rgb_max)) # 0 ~ 255
        seg_color = tuple(seg_color)
        while seg_color in self.used_colors:
            seg_color = np.uint8(np.random.uniform(rgb_low, rgb_max))
            seg_color = tuple(seg_color)
        self.used_colors.append(seg_color)

        return seg_color
    
    @staticmethod
    def create_segmented_object(obj_visibles, seg_texture, pr):
        seg_objects = []
        if not type(obj_visibles) == list:
            obj_visibles = [obj_visibles]
        for obj_visible in obj_visibles:
            seg_object = obj_visible.copy()
            seg_object.set_color([1, 1, 1])
            try:
                seg_object.remove_texture()
            except:
                pass
            seg_object.set_texture(texture=seg_texture,
                                decal_mode=True,
                                mapping_mode=TextureMappingMode(3),
                                uv_scaling=[1, 1],
                                repeat_along_u=True,
                                repeat_along_v=True,
                                )
            seg_object.set_renderable(False)
            seg_object.set_parent(obj_visible)
            seg_objects.append(seg_object)
        pr.step()
        if len(seg_objects) == 1:
            return seg_objects[0]
        else:
            return seg_objects

    def set_seg_state(self, seg):
        for seg_obj in self.seg_objects:
            seg_obj.set_renderable(seg)

    def reset(self):
        self.used_colors = []
