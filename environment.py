from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.const import ObjectType, TextureMappingMode, PrimitiveShape
from pyrep.backend import sim
from pyrep.objects.dummy import Dummy

import os
from os.path import join
import random
import numpy as np
from enum import Enum

from sensor.camera_manager import CameraManager, CameraType
from sensor.light import Light
from robot.mypanda import MyPanda

from externApi.fileApi import *
from externApi.imageProcessApi import *

# path
class TextureType(Enum):
    gray_texture = "./textures/gray_textures/"
    wood_texture = "./textures/wood_textures/"
    crawled_texture = "./textures/crawled_textures/"
    mixed_texture = "./textures/texture_mix_randomized_small/" #TODO:jsjs

class ArrangeType(Enum):
    align = 0
            
shape_type = [PrimitiveShape.CUBOID,
              PrimitiveShape.SPHERE,
              PrimitiveShape.CYLINDER,
              PrimitiveShape.CONE,]

class DREnv(object):
    
    def __init__(self, logger,
                       headless=False, 
                       dataset_ver=14,
                       process_id=0,
                       ):
        """
        scene (*.ttt) 

        """
        # logger
        self.logger = logger
        self.logger.info("Start initialize Scene")

        self.process_id = process_id
        self.pr = PyRep()

        self.logger.info(f"Dataset version is {dataset_ver}")
        if dataset_ver == 14:
            # light randomize config
            self._max_n_light = 3 # the number of point lights 2 ~ 3
            # camera randomize config
            self._camera_range = [(0.4, -0.2, 1.5), (0.9, 0.2, 1.8)] # relative to Table_top
            self.camera_type = CameraType.Zivid_ML
            # workspace config
            self._workspace_range = [(-0.4, -0.3), (0.4, 0.3)] # furniture (x, y) value based on self.workspace
            self._table_distractor_range = [(-0.55, -0.4), (0.55, 0.4)] # distractor (x, y) value based on self.workspace
            self._light_range = [(-4, -4), (4, 4)]
            # robot number
            self._robot_num = 1
            #TODO: stochastic config
            self._fn_occ = 0.5
            self._fn_flip_val = 0.5
            self._robot_randomize_val = 0 # np.random.rand() > self._robot_randomize_val: -> pass
            self._distractor_outer = 0.7
            # furniture texture color reference
            self._fn_color = [0.9, 0.9, 0.9] # white
            self._wood_color = [0.9, 0.7, 0.5]

        elif dataset_ver == 15:
            # light randomize config
            self._max_n_light = 3 # the number of point lights 2 ~ 3
            # camera randomize config
            self._camera_range = [(-0.05, 0.5, 1), (0.05, 0.65, 1.4)] # relative to self.workspace
            self.camera_type = CameraType.Azure
            # workspace config
            self._workspace_range = [(-0.4, -0.3), (0.4, 0.3)] # furniture (x, y) value based on self.workspace
            self._table_distractor_range = [(-0.55, -0.4), (0.55, 0.4)] # distractor (x, y) value based on self.workspace
            self._light_range = [(-4, -4), (4, 4)]
            # robot number
            self._gripper_robot_num = 1
            self._no_gripper_robot_num = 2
            #TODO: stochastic config
            self._fn_occ = 0.5
            self._fn_flip_val = 0.5
            self._robot_randomize_val = 0.1 # np.random.rand() > self._robot_randomize_val: -> pass
            self._distractor_outer = 0.7
            # furniture texture color reference
            self._fn_color = [0.9, 0.9, 0.9] # white
            self._wood_color = [0.9, 0.7, 0.5]
        
        elif dataset_ver == 16:
            # light randomize config
            self._max_n_light = 3 # the number of point lights 2 ~ 3
            # camera randomize config
            self._camera_range = [(-0.05, 0.5, 1), (0.05, 0.65, 1.4)] # relative to self.workspace
            self.camera_type = CameraType.Azure
            # workspace config
            self._workspace_range = [(-0.5, -0.3), (0.5, 0.3)] # furniture (x, y) value based on self.workspace
            self._table_distractor_range = [(-0.55, -0.4), (0.55, 0.4)] # distractor (x, y) value based on self.workspace
            self._light_range = [(-4, -4), (4, 4)]
            # robot number
            self._gripper_robot_num = 1
            self._no_gripper_robot_num = 2
            #TODO: stochastic config
            self._fn_occ = 0.5
            self._fn_flip_val = 0.5
            self._robot_randomize_val = 0.1 # np.random.rand() > self._robot_randomize_val: -> pass
            self._distractor_outer = 0.7
            # furniture texture color reference
            self._fn_color = [0.95, 0.95, 0.95] # white
            self._wood_color = [0.9, 0.7, 0.5]
        
        elif dataset_ver == 17:
            # light randomize config
            self._max_n_light = 3 # the number of point lights 2 ~ 3
            # camera randomize config
            self._camera_range = [(-0.05, 0.5, 1), (0.05, 0.65, 1.4)] # relative to self.workspace
            self.camera_type = CameraType.Azure
            # workspace config
            self._workspace_range = [(-0.4, -0.2), (0.4, 0.2)] # furniture (x, y) value based on self.workspace
            self._table_distractor_range = [(-0.55, -0.4), (0.55, 0.4)] # distractor (x, y) value based on self.workspace
            self._light_range = [(-4, -4), (4, 4)]
            # robot number
            self._gripper_robot_num = 1
            self._no_gripper_robot_num = 2
            #TODO: stochastic config
            self._fn_occ = 0.5
            self._fn_flip_val = 0.5
            self._robot_randomize_val = 0.1 # np.random.rand() > self._robot_randomize_val: -> pass
            self._distractor_outer = 0.7
            # furniture texture color reference
            self._fn_color = [0.95, 0.95, 0.95] # white
            self._wood_color = [0.9, 0.7, 0.5]
        
        else:
            print("error")
            exit()
            
        # static scene
        self.scene = "./scene/assembly_env_Azure"
        self._initialize_scene(self.scene, headless=headless)

        # get scene objects
        self.workspace = Shape("workspace_B")
        self.table = Shape("Table")
        self.table_top = Dummy("Table_top")
        self.light_base = Dummy("LightController")
        self.lights = [Light("Light1"), Light("Light2"),
                       Light("Light3"), Light("Light4")]
        self.floors = [Shape("Floor1"), Shape("Floor2"), Shape("Floor3"), Shape("Floor4"), Shape("Floor5")]
        self.collision_objects = [obj for obj in self.table.get_objects_in_tree(ObjectType.SHAPE, exclude_base=True) if "invisible" in obj.get_name()]

        asm_bases = Dummy("assembly_parts").get_objects_in_tree(ObjectType.SHAPE, exclude_base=True, first_generation_only=True)
        self.assembly_parts = [AssemblePart(asm_base) for asm_base in asm_bases]

        self.scene_furniture = []
        self.scene_assembled = []


        # align all furniture and assemblys (max height == 0.18)
        self.furnitures = [Furniture("Ikea_stefan_bottom"),
                           Furniture("Ikea_stefan_long"),
                           Furniture("Ikea_stefan_middle"),
                           Furniture("Ikea_stefan_short"),
                           Furniture("Ikea_stefan_side_left"),
                           Furniture("Ikea_stefan_side_right")]
        for fn in self.furnitures + self.assembly_parts:
            height, axis = self.get_rotation_axis(fn.respondable)
            fn.set_rotation_axis(height, axis)
        
        self.logger.debug("End initialize Scene")

        # texture
        self.logger.debug("initialize texture manager")
        self.texture_manager = TextureManager(self.process_id, self.pr)
        
        # camera setting
        self.camera_manager = CameraManager(self.camera_type, self.pr)
        
        # robot
        self.logger.debug("initialize robot")
        self.robot = Robot(self._gripper_robot_num, self._no_gripper_robot_num)
        self.collision_objects += self.robot.respondable

        # distractor 1~5
        self.distractor_bases = [Shape("workspace_C"), Shape("workspace_D")]
        self._distractor_range = [(-0.2, -2.5), (0.2, 2.5)]
        self.distractor = []  

        # box
        self.boxes = [Shape("box_A"), Shape("box_B")]
        self.scene_boxes = []
        self.min_box_num, self.max_box_num = 3, 6
        self._box_color = [0.99, 0.99, 0.99]
        self.segmentation_dummy = Dummy("segmentation")

        # occlusion
        self._is_occ = False

        # opposed side
        self._opposed_mode = False

    def _initialize_scene(self, scene_file, headless):
        if type(scene_file) == str:
            scene_file = scene_file + ".ttt"
            self.pr.launch(scene_file=scene_file, headless=headless)
        else:
            self.pr.launch(headless=headless)
        self.pr.start()
        
    #region domain randomize
    #region distractor
    def _create_distractors(self, num, is_table):
        distractors = []
        if is_table:
            scale = 0.04
            xy_position=self.workspace.get_position()[:2]
            for i in range(num):
                sh_type = random.choice(shape_type)
                size = list((np.random.rand(3) + 1) * scale)
                position = xy_position + [0]
                dis_sh = Shape.create(type=sh_type, 
                                      size=size,
                                      position=position,
                                      respondable=True,
                                      static=True)
                distractors.append(dis_sh)
                dis_sh.set_collidable(True)
            self.pr.step()
        else:
            scale = 1
            xy_position=self.distractor_bases[0].get_position()[:2]
            for i in range(num):
                sh_type = random.choice(shape_type)
                size = list((np.random.rand(3) + 1) * scale)
                position = xy_position + [size[2]]
                dis_sh = Shape.create(type=sh_type, 
                                      size=size,
                                      position=position,
                                      respondable=False,
                                      static=True)
                distractors.append(dis_sh)
            self.pr.step()
        return distractors
                
    def _randomize_distractor(self, num, is_table):
        self.logger.debug("start to randomize distractor")
        distractors = self._create_distractors(num, is_table)
        if is_table:
            distractor_range = self._table_distractor_range
            base = self.table_top
            for dis_sh in distractors:
                random_position = list(np.random.uniform(distractor_range[0], distractor_range[1]))
                random_position += [0.02]
                if np.random.rand() < self._distractor_outer:
                    axis_idx = np.random.randint(0,2)
                    val_idx = np.random.randint(0,2)
                    random_position[axis_idx] = distractor_range[val_idx][axis_idx]
                dis_sh.set_position(random_position, relative_to=base)
                random_orientation = list(np.random.rand(3))
                dis_sh.set_orientation(random_orientation)
                self.texture_manager.set_random_texture(dis_sh, TextureType.gray_texture)
        else:
            distractor_range = self._distractor_range
            for dis_sh in distractors:
                base = random.choice(self.distractor_bases)
                random_position = list(np.random.uniform(distractor_range[0], distractor_range[1]))
                random_position += [np.random.rand()*0.1]
                dis_sh.set_position(random_position, relative_to=base)
                random_orientation = list(np.random.rand(3))
                dis_sh.set_orientation(random_orientation)
                self.texture_manager.set_random_texture(dis_sh, TextureType.gray_texture)
        self.pr.step()
        self.distractor += distractors
        self.logger.debug("end to randomize distractor")
    #endregion

    #region scene randomize
    def _randomize_boxes(self):
        num = np.random.randint(self.min_box_num, self.max_box_num + 1)
        for i in range(num):
            box = random.choice(self.boxes)
            cp_box = box.copy()
            cp_box.set_parent(self.segmentation_dummy)
            self.scene_boxes.append(cp_box)        

        for box in self.scene_boxes:
            collision_state = True
            count = 0

            while collision_state and count < 5:
                position = list(np.random.uniform(self._workspace_range[0], self._workspace_range[1]))
                position += [0.08]
                rand_ori = [0, 0] + [np.random.rand()]
                box.set_position(position, relative_to=self.workspace)
                box.set_orientation(rand_ori)
                self.pr.step()    
                collision_state = self._check_collision(box)
                count += 1

            self.set_auxiliary_color(box, [0,0,1])
            self.texture_manager.set_random_texture(box, TextureType.gray_texture, refer=self._box_color)

    def _randomize_light(self):
        for light in self.lights:
            light.light_off()
        self.n_light = np.random.randint(1, self._max_n_light + 1)
        sampled_light = random.sample(self.lights, self.n_light)
        for light in sampled_light:
            random_position = list(np.random.uniform(self._light_range[0], self._light_range[1]))
            random_position += [0]
            light.set_position(random_position, relative_to=self.light_base)
            random_diffuse = list(0.5 + np.random.rand(3)*0.5)
            random_specular = list(0.5 + np.random.rand(3)*0.5)
            light.light_on(random_diffuse,random_specular)

    def _randomize_floor(self):
        bottom = self.floors[0]
        walls = self.floors[1:]
        self.texture_manager.set_random_texture(bottom, TextureType.gray_texture)
        self.texture_manager.set_random_texture(walls, TextureType.gray_texture)
    
    def _randomize_table(self):
        table_texture = self.texture_manager._get_randomize_texture(TextureType.mixed_texture)
        self.table.set_texture(texture=table_texture,
                               mapping_mode=TextureMappingMode(3),
                               uv_scaling=[3, 3],
                               repeat_along_u=True,
                               repeat_along_v=True)
    
    def _randomize_robot(self):
        # randomize pose
        self.robot.randomize_pose()
        # randomize texture
        for vis in self.robot.visible:
            if np.random.rand() > self._robot_randomize_val:
                continue
            self.texture_manager.set_random_texture(vis, TextureType.gray_texture)
        self.pr.step()
    
    #endregion

    #region furniture randomize
    def _sample_furnitures(self, asm_num, fn_num):
        # if self._is_occ:
        #     for fn in self.furnitures:
        #         fn.switch_respondable_to_bbox(is_bbox=False)
        # else:
        #     for fn in self.furnitures:
        #         fn.switch_respondable_to_bbox(is_bbox=True)
        # self.pr.step()
        for asm in self.assembly_parts:
            asm.deactivate()
        for fn in self.furnitures:
            fn.activate(False)
        self.pr.step()
        for asm in self.assembly_parts:
            asm.reset()
        for fn in self.furnitures:
            fn.reset()
        
        self.scene_assembled = random.sample(self.assembly_parts, asm_num)
        #TODO: if impossible to assemble
        for asm in self.scene_assembled:
            _ = asm.activate(self.furnitures)
        
        unused_fns = []
        for fn in self.furnitures:
            if fn.is_assembled:
                continue
            unused_fns.append(fn)
        self.scene_furniture = random.sample(unused_fns, fn_num)
        for fn in self.scene_furniture:
            fn.activate(True)
    


    def _randomize_furniture_position(self, fn):
        """
        randomize furniture position in workspace
        fn can be primitive or assembled furniture
        """
        height = fn.get_height(relative_to=self.workspace)
        random_position = list(np.random.uniform(self._workspace_range[0], self._workspace_range[1]))
        random_position += [-1 * height]
        
        fn.set_position(random_position, relative_to=self.workspace)
        self.pr.step()
    
    def _randomize_furniture_rotation(self, fn):
        
        # rotate along rot_axis
        random_rot = [0, 0, 0]
        random_rot[fn.rot_axis] = np.random.rand() * np.pi * 2
        # flip
        if np.random.rand(1) < self._fn_flip_val:
            flip_idx = (fn.rot_axis + 1) % 3
            random_rot[flip_idx] = np.pi            
        else:
            pass
        
        # random rotate
        random_rot = np.random.rand(3) * np.pi * 2
        
        fn.rotate(random_rot)
        self.pr.step()

    def _randomize_furniture_pose(self, fn):
        fn.set_respondable(False)
        self.pr.step()
        collision_state = True
        count = 0
        while collision_state and count < 5:
            self._randomize_furniture_rotation(fn)
            self._randomize_furniture_position(fn)
            
            collision_state = self._check_collision(fn.respondable)
            count += 1
        print("-"*20)
        if collision_state:
            current_pos = fn.get_position()
            current_pos[2] += 10
            fn.set_position(current_pos)
        fn.set_respondable(True)
        self.pr.step()

    def _randomize_furniture_texture(self, fn):
        rand_visible = [fn.visible_part]
        wood_visible = [fn.wood]
        if fn.is_hole:
            wood_visible += [fn.hole]
        
        self.texture_manager.set_random_texture(rand_visible, TextureType.gray_texture, refer=self._fn_color)
        self.texture_manager.set_random_texture(wood_visible, TextureType.wood_texture)
        
    def _randomize_furniture(self):
        self.logger.debug("start to randomize furniture")
        for asm in self.scene_assembled:
            self._randomize_furniture_pose(asm)
            for fn in asm.used_fn:
                self._randomize_furniture_texture(fn)
        for fn in self.scene_furniture:
            self._randomize_furniture_pose(fn)
            self._randomize_furniture_texture(fn)
        self.pr.step()
        self.logger.debug("end to randomize furniture")

    def _check_collision(self, obj):
        #TODO:
        is_collision = False
        for c_obj in self.collision_objects + self.distractor + self.boxes:
            if c_obj.check_collision(obj):
                is_collision = True
                print(c_obj.get_name())
                return is_collision
        
        for f_obj in self.scene_furniture + self.scene_assembled:
            res = f_obj.respondable
            if res == obj:
                continue
            if res.check_collision(obj):
                is_collision = True
                print(res.get_name())
                return is_collision

        return is_collision

    def get_rotation_axis(self, obj):
        bbox = obj.get_bounding_box()
        xyz = [bbox[1], bbox[3], bbox[5]]
        # find align info
        obj.set_orientation([0, 0, 0])
        self.pr.step()
        val, idx = min((val, idx) for (idx, val) in enumerate(xyz))
        if idx == 0: # x is min
            obj.rotate([0, np.pi / 2, 0])
        elif idx == 1: # y is min
            obj.rotate([np.pi / 2, 0, 0])
        else: # z is min
            pass
        self.pr.step()

        return val, idx
    @staticmethod
    def set_auxiliary_color(obj: Shape, color) -> None:
        """Sets the color of the shape.

        :param color: The r, g, b values of the shape.
        :return:
        """
        sim.simSetShapeColor(
            obj._handle, None, sim.sim_colorcomponent_auxiliary, color)
    
    #endregion

    def _randomize_camera(self):
        self.logger.debug("start randomize camera pose")
        # randomize view point
        defalut_view_point = self.workspace
        self.camera_manager.set_rotation_base_to(defalut_view_point)

        # randomize position
        random_position = list(np.random.uniform(self._camera_range[0], self._camera_range[1]))
        if self._opposed_mode:
            random_position[1] *= -1
        
        self.camera_manager.set_position(random_position, self.workspace)

        # randomize perspective angle
        
        self.camera_manager.set_random_angle()

        self.logger.debug("end randomize camera pose")

    #endregion
    def set_stable(self):
        for fn in self.scene_furniture + self.scene_assembled:
            fn.set_dynamic(True)
        self.pr.step()
        for i in range(3):
            self.pr.step()
        for fn in self.scene_furniture + self.scene_assembled:
            fn.set_dynamic(False)
        self.pr.step()

    def randomize_scene(self):
        self.logger.debug("start to randomize scene")
        self._randomize_floor()
        self._randomize_light()
        self._randomize_table()
        self._randomize_robot()
        self.pr.step()
        self.logger.debug("end randomize scene")

    def reset(self, assembly_num, furniture_num):
        self.logger.info("start to reset scene")
        self.texture_manager.reset()
        self.camera_manager.reset()
        self.camera_manager.set_activate(False)

        for distractor in self.distractor:
            distractor.remove()
        self.distractor = []

        for box in self.scene_boxes:
            box.remove()
        self.scene_boxes = []

        # if opposed_mode => camera location is opposed
        if np.random.rand() < 0.5:
            self._opposed_mode = True
        else:
            self._opposed_mode = False

        self._randomize_distractor(5, is_table=True)
        self._randomize_distractor(5, is_table=False)
        self._randomize_boxes()

        self._sample_furnitures(assembly_num, furniture_num)
        if np.random.rand() < self._fn_occ and len(self.scene_furniture) > 1:
            self._is_occ = True
        else:
            self._is_occ = False
        
        self._randomize_furniture()
        self.randomize_scene()
               
        self.camera_manager.set_activate(True)
        self.set_stable()
        self.logger.info("End to reset scene")

    def test_step(self, assembly_num, furniture_num):
        self.texture_manager.reset()
        self._sample_furnitures(assembly_num, furniture_num)
        if np.random.rand() < self._fn_occ and len(self.scene_furniture) > 1:
            self._is_occ = True
        else:
            self._is_occ = False
        count = 0
        for i in range(100):
            if count % 10 == 0:
                self._randomize_furniture()
            self.pr.step()
            count += 1
            
    def step(self):
        """
        1. randomize furniture pose
        2. randomize camera pose
        3. (opt)randomize robot pose
        """
        #TODO: randomize furniture pose
        fn = random.choice(self.scene_assembled + self.scene_furniture)
        self._randomize_furniture_pose(fn)
        self.logger.debug("start one step")
        self._randomize_camera()
        self.set_stable()
        self.logger.debug("end one step")

    def camera_test(self):
        defalut_view_point = self.workspace
        self.camera_manager.set_rotation_base_to(defalut_view_point)
        
        for p_angle in np.linspace(self._pangle_range[0], self._pangle_range[1], 10):
            for pos in np.linspace(self._camera_range[0], self._camera_range[1], 10):
                self.camera_manager.set_perspective_angle(p_angle)
                self.camera_manager.set_position(pos, self.table_top)
                for i in range(10):
                    self.pr.step()

    def test_align(self):
        for i in range(100):
            fn = random.choice(self.scene_furniture+self.scene_assembled)
            self._randomize_furniture_pose(fn)

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def get_images(self):
        self.logger.debug("start to get image")
        self.camera_manager.capture()

        return self.camera_manager.get_images()

class SceneObject(object):
    def __init__(self, scene_obj, class_id=-1):
        self.visible = scene_obj
        self.respondable = self.visible
        self.class_id = class_id
        self.set_dynamic(False)
        self.set_respondable(False)
        self.set_renderable(True)

    def _extract_respondable_part(self):
        try:
            respondable = self.visible.get_convex_decomposition()
        except:
            respondable = self.visible.copy()
        respondable.set_renderable(False)
        self._is_extracted = True
        
        return respondable

    def set_respondable(self, is_respondable):
        if self.visible == self.respondable and is_respondable:
            self.respondable = self._extract_respondable_part()
            self.visible.set_parent(self.respondable)
        self.respondable.set_respondable(is_respondable)
        self._is_respondable = is_respondable

    def set_dynamic(self, is_dynamic):
        self.respondable.set_dynamic(is_dynamic)
        self._is_dynamic = is_dynamic
    
    def set_renderable(self, is_renderable):
        self.visible.set_renderable(is_renderable)
        self._is_renderable = is_renderable

    def set_detectable(self, is_detectable):
        self.visible.set_detectable(is_detectable)
        self._is_detectable = is_detectable

    def set_collidable(self, is_collidable):
        self.respondable.set_dynamic(is_collidable)
        self._is_dynamic = is_collidable

    def set_position(self, position, relative_to=None):
        self.respondable.set_position(position, relative_to)
    def get_position(self, relative_to=None):
        return self.respondable.get_position(relative_to)
    
    def set_orientation(self, orientation, relative_to=None):
        self.respondable.set_orientation(orientation, relative_to)
    def get_orientation(self, relative_to=None):
        return self.respondable.get_orientation(relative_to)
    
    def set_pose(self, pose, relative_to=None):
        self.respondable.set_pose(pose, relative_to)
    def get_pose(self, relative_to=None):
        self.respondable.get_pose(relative_to)

    def rotate(self, rotation):
        self.respondable.rotate(rotation)

    def set_parent(self, parent):
        self.respondable.set_parent(parent)

    def set_rand_texture(self, texture):
        mapping_ind = np.random.randint(4)
        uv_scale = list(np.random.uniform((0, 0), (5, 5)))
        self.visible.set_texture(texture=texture,
                                 mapping_mode=TextureMappingMode(mapping_ind),
                                 uv_scaling=uv_scale,
                                 repeat_along_u=True,
                                 repeat_along_v=True)  

    def remove(self):
        self.visible.remove()
        if self._is_extracted:
            self.respondable.remove()

    def rotate_rel(self, rotation, relative_to):
        relto = relative_to
        M = self.respondable.get_matrix()
        m = relto.get_matrix()
        x_axis = [m[0], m[4], m[8]]
        y_axis = [m[1], m[5], m[9]]
        z_axis = [m[2], m[6], m[10]]
        pos = [m[3], m[7], m[11]]
        M = sim.simRotateAroundAxis(M, z_axis, pos, rotation[2])
        M = sim.simRotateAroundAxis(M, y_axis, pos, rotation[1])
        M = sim.simRotateAroundAxis(M, x_axis, pos, rotation[0])
        self.respondable.set_matrix(M)
    
    def check_collision(self, obj):
        self.respondable.check_collision(obj)

    @staticmethod
    def _create_by_CAD(filepath, scaling_factor, class_id, is_texture=False, is_dynamic=True):
        if is_texture:
            try:
                obj = Shape.import_shape(filename=filepath,
                                         scaling_factor=scaling_factor,)          
            except:
                print("error")
        else:
            try:
                obj = Shape.import_mesh(filename=filepath,
                                        scaling_factor=scaling_factor,)
            except:
                print("error")
        
        return SceneObject(obj, class_id)

class Furniture(object):
    """
    respondable_part
    visible_parts
    wood_parts
    hole_parts
    """
    #TODO:hole mask 

    def __init__(self, furniture_name):
        self.name = furniture_name
        self.respondable = Shape(furniture_name)
        self.respondable.set_mass(0.01)
        self.temp_res = self.respondable
        self.visible_part = Shape(furniture_name + "_visible_tex")
        self.wood = Shape(furniture_name + "_visible_wood")
        try:
            self.hole = Shape(furniture_name + "_hole_visible")
            self.is_hole = True
        except:
            self.hole = None
            self.is_hole = False
        self.is_activate = False
        self._init_parent = self.respondable.get_parent()
        self.is_assembled = False
        self._initialize_bbox()

    def _initialize_bbox(self):
        bbox = self.respondable.get_bounding_box()
        ori = self.respondable.get_orientation()
        position = self.respondable.get_position()
        xyz = [bbox[1], bbox[3], bbox[5]]
        size = list(np.array(xyz)*2)
        self.bbox = Shape.create(type=PrimitiveShape.CUBOID,
                                 size=size,
                                 respondable=False,
                                 renderable=False,
                                 static=False,
                                 position=position,
                                 orientation=ori,
                                )
        self.bbox.set_detectable(False)
        
    def get_height(self, relative_to=None):
        """get current height relative to world coordinate
        """
        bbox = self.respondable.get_bounding_box()
        m = self.respondable.get_matrix()
        bbox_axis = [
            [m[0], m[4], m[8]], # x 
            [m[1], m[5], m[9]], # y
            [m[2], m[6], m[10]] # z
        ]
        bbox_axis = np.array(bbox_axis)
        object_pos = np.array(self.respondable.get_position(relative_to=relative_to))
        min_height = np.inf
        for i in range(8):
            delta = np.array([bbox[1], bbox[3], bbox[5]])
            if i // 4 == 0: 
                pass
            else: # 4,5,6,7
                delta[0] *= -1
            if (i % 4) // 2 > 0: # 2, 3, 6, 8
                delta[1] *= -1
            else: 
                pass                
            if i % 2 == 0:
                pass
            else: # 1, 3, 5, 7
                delta[2] *= -1
            pos = np.dot(delta, bbox_axis)
            if pos[2] < min_height:
                min_height = pos[2]

        return min_height

    def switch_respondable_to_bbox(self, is_bbox):
        if is_bbox:
            self.respondable = self.bbox
            self.temp_res.set_parent(self.bbox)
            
        else:
            self.respondable = self.temp_res
            self.respondable.set_parent(None)

    def set_rotation_axis(self, height, rot_axis):
        self.height = height
        self.rot_axis = rot_axis
        self._initial_pose = self.respondable.get_pose()

    def set_position(self, position, relative_to=None):
        self.respondable.set_position(position, relative_to)
    def get_position(self, relative_to=None):
        return self.respondable.get_position(relative_to)
    
    def set_orientation(self, orientation, relative_to=None):
        self.respondable.set_orientation(orientation, relative_to)
    def get_orientation(self, relative_to=None):
        return self.respondable.get_orientation(relative_to)
    
    def set_pose(self, pose, relative_to=None):
        self.respondable.set_pose(pose, relative_to)
    def get_pose(self, relative_to=None):
        self.respondable.get_pose(relative_to)

    def set_respondable(self, is_respondable):
        self.respondable.set_respondable(is_respondable)

    def rotate(self, rotation):
        self.respondable.rotate(rotation)

    def set_parent(self, parent):
        self.respondable.set_parent(parent)

    def check_collision(self, obj):
        self.respondable.check_collision(obj)

    def set_dynamic(self ,is_dynamic):
        self.respondable.set_dynamic(is_dynamic)

    def activate(self, is_activate):
        self.respondable.set_respondable(is_activate)
        self.visible_part.set_renderable(is_activate)
        self.wood.set_renderable(is_activate)
        if self.is_hole:
            self.hole.set_renderable(is_activate)

    def reset(self):
        self.respondable.set_pose(self._initial_pose)

    def set_assembled(self, is_assembled):
        self.activate(is_assembled)
        self.respondable.set_dynamic(not is_assembled)
        self.respondable.set_respondable(not is_assembled)
        self.respondable.set_collidable(not is_assembled)
        if not is_assembled:
            self.respondable.set_pose(self._initial_pose)
            self.set_parent(self._init_parent)
        self.is_assembled = is_assembled
        
class AssemblePart(object):
    """assembled furnitures
    assembled part
    part1
    part2
    ...

    """
    def __init__(self, assembled_part: Shape):
        self.respondable = assembled_part
        self.respondable.set_mass(0.01)
        self.sub_parts = self._get_sub_parts()
        self.used_fn = []

    def get_height(self, relative_to=None):
        """get current height relative to world coordinate
        """
        bbox = self.respondable.get_bounding_box()
        m = self.respondable.get_matrix()
        bbox_axis = [
            [m[0], m[4], m[8]], # x 
            [m[1], m[5], m[9]], # y
            [m[2], m[6], m[10]] # z
        ]
        bbox_axis = np.array(bbox_axis)
        object_pos = np.array(self.respondable.get_position(relative_to=relative_to))
        min_height = np.inf
        for i in range(8):
            delta = np.array([bbox[1], bbox[3], bbox[5]])
            if i // 4 == 0: 
                pass
            else: # 4,5,6,7
                delta[0] *= -1
            if (i % 4) // 2 > 0: # 2, 3, 6, 8
                delta[1] *= -1
            else: 
                pass                
            if i % 2 == 0:
                pass
            else: # 1, 3, 5, 7
                delta[2] *= -1
            pos = np.dot(delta, bbox_axis)
            if pos[2] < min_height:
                min_height = pos[2]

        return min_height

    def get_bbox(self):
        bbox = self.respondable.get_bounding_box()
        xyz = [bbox[1], bbox[3], bbox[5]]
        xyz = list(np.array(xyz) *2)
        bbox = Shape.create(PrimitiveShape.CUBOID, size=xyz)
        bbox.set_position([0, 0, 0], relative_to=self.respondable)
        bbox.set_orientation([0, 0, 0], relative_to=self.respondable)
        bbox.set_renderable(False)
        bbox.set_detectable(False)

        return bbox
        
    def set_rotation_axis(self, height, rot_axis):
        self.height = height
        self.rot_axis = rot_axis
        self._initial_pose = self.respondable.get_pose()

    def set_position(self, position, relative_to=None):
        self.respondable.set_position(position, relative_to)
    def get_position(self, relative_to=None):
        return self.respondable.get_position(relative_to)
    
    def set_orientation(self, orientation, relative_to=None):
        self.respondable.set_orientation(orientation, relative_to)
    def get_orientation(self, relative_to=None):
        return self.respondable.get_orientation(relative_to)
    
    def set_pose(self, pose, relative_to=None):
        self.respondable.set_pose(pose, relative_to)
    def get_pose(self, relative_to=None):
        self.respondable.get_pose(relative_to)

    def set_respondable(self, is_respondable):
        self.respondable.set_respondable(is_respondable)

    def rotate(self, rotation):
        self.respondable.rotate(rotation)

    def _get_sub_parts(self):
        childs = self.respondable.get_objects_in_tree(ObjectType.SHAPE, exclude_base=True)
        sub_parts = []
        for child in childs:
            name = child.get_name()
            child_type = name.split("_")[-1]
            if child_type == "mask":
                continue
            sub_parts.append((child, child_type))

        return sub_parts

    def _catch_fn_parts(self, furnitures):
        #TODO: pose는 0000000 이면 안됨

        self.used_fn = []
        for part, pt_name in self.sub_parts:
            is_exist = False
            for fn in furnitures:
                if fn.is_assembled:
                    continue
                if pt_name in fn.name:
                    is_exist = True
                    fn.set_parent(self.respondable)
                    pose = part.get_pose(relative_to=self.respondable)
                    fn.set_pose(pose, relative_to=self.respondable)
                    fn.set_assembled(True)
                    self.used_fn.append(fn)
                    break
            if not is_exist:
                return False
        
        return True

    def release(self):
        for fn in self.used_fn:
            fn.set_assembled(False)
        self.used_fn = []

    def set_dynamic(self, is_dynamic):
        self.respondable.set_dynamic(is_dynamic)

    def deactivate(self):
        self.respondable.set_respondable(False)
        self.release()

    def reset(self):
        self.respondable.set_pose(self._initial_pose)   

    def activate(self, furnitures):
        if self._catch_fn_parts(furnitures):
            self.respondable.set_respondable(True)
            return True
        else:
            return False
        
class Robot(object):
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

    def reset(self):
        for robot in self.robots:
            robot.reset()

class TextureManager(object):
    def __init__(self, process_id, pr):
        self.process_id = process_id
        self.pr = pr
        self._color_variance = 0.1
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
    
    #TODO:jsjs
    def _get_mixed_randomized_teuxture(self, texture_path):
        rand_texture_path = texture_path
        return rand_texture_path

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
    
    def _get_randomize_texture(self, texture_type, refer=None):
        rand_texture_path = self._get_random_texture(texture_type)

        if texture_type == TextureType.gray_texture:
            rand_texture_path = self._get_grad_randomized_texture(rand_texture_path, refer=refer)
        elif texture_type == TextureType.mixed_texture:
            rand_texture_path = self._get_mixed_randomized_teuxture(rand_texture_path)#TODO:jsjs

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
        texture = self._get_randomize_texture(texture_type, refer=refer)
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
