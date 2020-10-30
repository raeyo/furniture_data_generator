from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.const import ObjectType, TextureMappingMode, PrimitiveShape
from pyrep.backend import sim
from pyrep.objects.dummy import Dummy

import os
import random
import numpy as np

from sensor.camera_manager import CameraManager, CameraType
from sensor.light import Light

from externApi.fileApi import *
from externApi.imageProcessApi import *

from environment_manager import *
from scene_object import *
from const import *

from timeout import timeout

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
            # stochastic config
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
            # stochastic config
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
            # stochastic config
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
            # stochastic config
            self._fn_occ = 0.5
            self._fn_flip_val = 0.5
            self._robot_randomize_val = 0.1 # np.random.rand() > self._robot_randomize_val: -> pass
            self._distractor_outer = 0.7
            # furniture texture color reference
            self._fn_color = [0.95, 0.95, 0.95] # white
            self._wood_color = [0.9, 0.7, 0.5]
        
        elif dataset_ver == 18: # for bottom
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
            # stochastic config
            self._fn_occ = 0.5
            self._fn_flip_val = 0.5
            self._robot_randomize_val = 0.1 # np.random.rand() > self._robot_randomize_val: -> pass
            self._distractor_outer = 0.7
            # furniture texture color reference
            self._fn_color = [0.95, 0.95, 0.95] # white
            self._wood_color = [0.9, 0.7, 0.5]

        elif dataset_ver == 19: # randomize camera pose
            # light randomize config
            self._max_n_light = 3 # the number of point lights 2 ~ 3
            # camera randomize config
            self._camera_range = [(-0.9, 0.5, 0.1), (0.9, 0.65, 1.4)] # relative to self.workspace
            self.camera_type = CameraType.Azure
            # workspace config
            self._workspace_range = [(-0.4, -0.2), (0.4, 0.2)] # furniture (x, y) value based on self.workspace
            self._table_distractor_range = [(-0.55, -0.4), (0.55, 0.4)] # distractor (x, y) value based on self.workspace
            self._light_range = [(-4, -4), (4, 4)]
            # robot number
            self._gripper_robot_num = 1
            self._no_gripper_robot_num = 2
            # stochastic config
            self._fn_occ = 0.5
            self._fn_flip_val = 0.5
            self._robot_randomize_val = 0.1 # np.random.rand() > self._robot_randomize_val: -> pass
            self._distractor_outer = 0.7
            # furniture texture color reference
            self._fn_color = [0.95, 0.95, 0.95] # white
            self._wood_color = [0.9, 0.7, 0.5]
        
        elif dataset_ver == 20:
            """
            1. camera z: 0.1 -> 0.2
            2. add 1 part data
            3. fix box z position
            """ 
            # light randomize config
            self._max_n_light = 3 # the number of point lights 2 ~ 3
            # camera randomize config
            self._camera_range = [(-0.9, 0.5, 0.1), (0.9, 0.65, 1.4)] # relative to self.workspace
            self.camera_type = CameraType.Azure
            # workspace config
            self._workspace_range = [(-0.4, -0.2), (0.4, 0.2)] # furniture (x, y) value based on self.workspace
            self._table_distractor_range = [(-0.55, -0.4), (0.55, 0.4)] # distractor (x, y) value based on self.workspace
            self._light_range = [(-4, -4), (4, 4)]
            # robot number
            self._gripper_robot_num = 1
            self._no_gripper_robot_num = 2
            # stochastic config
            self._fn_occ = 0.5
            self._fn_flip_val = 0.5
            self._robot_randomize_val = 0.1 # np.random.rand() > self._robot_randomize_val: -> pass
            self._distractor_outer = 0.7
            # furniture texture color reference
            self._fn_color = [0.95, 0.95, 0.95] # white
            self._wood_color = [0.9, 0.7, 0.5]
        
        elif dataset_ver == 21:
            """
            1. camera z: 0.1 -> 0.2
            2. add 1 part data
            3. fix box z position
            """ 
            # light randomize config
            self._max_n_light = 3 # the number of point lights 2 ~ 3
            # camera randomize config
            self._camera_range = [(-0.8, 0.55, 0.2), (0.8, 0.7, 1.4)] # relative to self.workspace
            self.camera_type = CameraType.Azure
            # workspace config
            self._workspace_range = [(-0.4, -0.2), (0.4, 0.2)] # furniture (x, y) value based on self.workspace
            self._table_distractor_range = [(-0.55, -0.4), (0.55, 0.4)] # distractor (x, y) value based on self.workspace
            self._light_range = [(-4, -4), (4, 4)]
            # robot number
            self._gripper_robot_num = 1
            self._no_gripper_robot_num = 2
            # stochastic config
            self._fn_occ = 0.5
            self._fn_flip_val = 0.5
            self._robot_randomize_val = 0.1 # np.random.rand() > self._robot_randomize_val: -> pass
            self._distractor_outer = 0.7
            # furniture texture color reference
            self._fn_color = [0.95, 0.95, 0.95] # white
            self._wood_color = [0.9, 0.7, 0.5]

        elif dataset_ver == 22:
            """
            1. use same texture for all furniture
            2. table texture -> black
            
            """ 
            # light randomize config
            self._max_n_light = 3 # the number of point lights 2 ~ 3
            # camera randomize config
            self._camera_range = [(-0.8, 0.55, 0.2), (0.8, 0.7, 1.4)] # relative to self.workspace
            self.camera_type = CameraType.Azure
            # workspace config
            self._workspace_range = [(-0.4, -0.2), (0.4, 0.2)] # furniture (x, y) value based on self.workspace
            self._table_distractor_range = [(-0.55, -0.4), (0.55, 0.4)] # distractor (x, y) value based on self.workspace
            self._light_range = [(-4, -4), (4, 4)]
            # robot number
            self._gripper_robot_num = 1
            self._no_gripper_robot_num = 2
            # stochastic config
            self._fn_occ = 0.5
            self._fn_flip_val = 0.5
            self._robot_randomize_val = 0.1 # np.random.rand() > self._robot_randomize_val: -> pass
            self._distractor_outer = 0.7
            # furniture texture color reference
            self._fn_color = [0.95, 0.95, 0.95] # white
            self._wood_color = [0.9, 0.7, 0.5]
        
        elif dataset_ver == 23:
            """
            for grasping dataset
            """ 
            # light randomize config
            self._max_n_light = 3 # the number of point lights 2 ~ 3
            # camera randomize config
            self._camera_range = CameraLocation # related to world frame
            self.camera_type = CameraType.Azure
            # workspace config
            self._workspace_range = [(-0.4, -0.2), (0.4, 0.2)] # furniture (x, y) value based on self.workspace
            self._table_distractor_range = [(-0.55, -0.4), (0.55, 0.4)] # distractor (x, y) value based on self.workspace
            self._light_range = [(-4, -4), (4, 4)]
            # robot number
            self._gripper_robot_num = 3
            self._no_gripper_robot_num = 0
            # stochastic config
            self._fn_occ = 0.5
            self._fn_flip_val = 0.5
            self._robot_randomize_val = 0.1 # np.random.rand() > self._robot_randomize_val: -> pass
            self._distractor_outer = 0.7
            # furniture texture color reference
            self._fn_color = [0.95, 0.95, 0.95] # white
            self._wood_color = [0.9, 0.7, 0.5]
        
        elif dataset_ver == 24:
            """
            normal dataset
            use CameraLocation, ViewType to randomize camera
            """ 
            # light randomize config
            self._max_n_light = 3 # the number of point lights 2 ~ 3
            # camera randomize config
            self._camera_range = CameraLocation # related to world frame
            self.camera_type = CameraType.Azure
            # workspace config
            self._workspace_range = [(-0.4, -0.2), (0.4, 0.2)] # furniture (x, y) value based on self.workspace
            self._table_distractor_range = [(-0.55, -0.4), (0.55, 0.4)] # distractor (x, y) value based on self.workspace
            self._light_range = [(-4, -4), (4, 4)]
            # robot number
            self._gripper_robot_num = 3
            self._no_gripper_robot_num = 0
            # stochastic config
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
        self.scene = "./scene/assembly_env_SNU"
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
        self.min_asm_num, self.max_asm_num = np.inf, -np.inf
        for asm_part in self.assembly_parts:
            if asm_part.asm_num < self.min_asm_num:
                self.min_asm_num = asm_part.asm_num
            if asm_part.asm_num > self.max_asm_num:
                self.max_asm_num = asm_part.asm_num

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
        self.robot_manager = RobotManager(self._gripper_robot_num, self._no_gripper_robot_num)
        self.collision_objects += self.robot_manager.respondable

        # distractor 1~5
        self.distractor_bases = [Shape("workspace_C"), Shape("workspace_D")]
        self._distractor_range = [(-0.2, -2.5), (0.2, 2.5)]
        self.distractor = []  

        # box
        self.scene_boxes = [Shape("sub_table0"), Shape("sub_table1"), Shape("sub_table2"), Shape("sub_table3")]
        self._box_color = [0.99, 0.99, 0.99]
        self.segmentation_dummy = Dummy("segmentation")

        # opposed side
        self._opposed_mode = False

        # overfitting texture
        self.furniture_texture = self.texture_manager.get_randomize_texture(TextureType.gray_texture, refer=self._fn_color)
        self.wood_texture = self.texture_manager.get_randomize_texture(TextureType.wood_texture)
        self._table_color = [0, 0, 0]
        
        self.table_texture_type = random.choice(list(TableTextureType))
        self.table_texture = self.texture_manager.get_randomize_texture(self.table_texture_type.value, refer=self._table_color)

        # grasp workspace
        self.grasp_workspace = Shape("workspace_grasp")
        self.grasp_range = [(-0.2, -0.1), (0.2, 0.1)]
        
        # world frame
        self.world_frame = Dummy("world_frame")

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
                sh_type = random.choice(list(PrimitiveShape))
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
                sh_type = random.choice(list(PrimitiveShape))
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
        for box in self.scene_boxes:
            # collision_state = True
            # count = 0

            # while collision_state and count < 5:
            #     position = list(np.random.uniform(self._workspace_range[0], self._workspace_range[1]))
            #     position += [0.07 - 0.16]
            #     rand_ori = [0, 0] + [np.random.rand() * np.pi]
            #     box.set_position(position, relative_to=self.workspace)
            #     box.set_orientation(rand_ori)
            #     self.pr.step()
            #     collision_state = self._check_collision(box, is_box=True)
            #     count += 1
            # self.set_auxiliary_color(box, [0,0,1])
            # self.texture_manager.set_random_texture(box, TextureType.gray_texture, refer=self._box_color)
            self.texture_manager._set_random_texture(box, self.table_texture)

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
        # self.table_texture = self.texture_manager.get_randomize_texture(TextureType.mixed_texture)
        
        self.table.set_texture(texture=self.table_texture,
                               mapping_mode=TextureMappingMode(3),
                               uv_scaling=[3, 3],
                               repeat_along_u=True,
                               repeat_along_v=True)
    
    def _randomize_robot(self):
        # randomize pose
        self.robot_manager.randomize_pose()
        # randomize texture
        for vis in self.robot_manager.visible:
            if np.random.rand() > self._robot_randomize_val:
                continue
            self.texture_manager.set_random_texture(vis, TextureType.gray_texture)
        self.pr.step()
    
    #endregion

    #region furniture randomize
    def _sample_furnitures(self, asm_num, fn_num):
        for asm in self.assembly_parts:
            asm.deactivate()
        for fn in self.furnitures:
            fn.activate(False)
        self.pr.step()
        for asm in self.assembly_parts:
            asm.reset()
        for fn in self.furnitures:
            fn.reset()
        
        self.scene_assembled = []
        for i in range(asm_num):
            choosed_num = np.random.randint(self.min_asm_num, self.max_asm_num + 1)
            while True:
                choosed_part = random.choice(self.assembly_parts)
                if choosed_part.asm_num == choosed_num:
                    self.scene_assembled.append(choosed_part)
                    break
                else:
                    continue

        #TODO: if impossible to assemble
        for asm in self.scene_assembled:
            _ = asm.activate(self.furnitures)
        
        unused_fns = []
        for fn in self.furnitures:
            if fn.is_assembled:
                continue
            unused_fns.append(fn)
        if len(unused_fns) >= fn_num:
            self.scene_furniture = random.sample(unused_fns, fn_num)
        else:
            self.scene_furniture = unused_fns
        # self.scene_furniture = [self.furnitures[0]] # for bottom
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
        self.logger.debug("start to randomize pose furniture")
        fn.set_respondable(False)
        self.pr.step()
        collision_state = True
        count = 0
        while collision_state and count < 5:
            self._randomize_furniture_rotation(fn)
            self._randomize_furniture_position(fn)
            self.logger.debug(f"check furniture collision {count}")
            collision_state = self._check_collision(fn.respondable)
            count += 1
        if collision_state:
            current_pos = fn.get_position()
            current_pos[2] += 4
            fn.set_position(current_pos)
        fn.set_respondable(True)
        self.pr.step()

    def _randomize_furniture_texture(self, fn):
        rand_visible = [fn.visible_part]
        wood_visible = [fn.wood]
        if fn.is_hole:
            wood_visible += [fn.hole]
        
        # self.texture_manager.set_random_texture(rand_visible, TextureType.gray_texture, refer=self._fn_color)
        # self.texture_manager.set_random_texture(wood_visible, TextureType.wood_texture)
        self.texture_manager._set_random_texture(rand_visible, self.furniture_texture)
        self.texture_manager._set_random_texture(wood_visible, self.wood_texture)
        
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

    @timeout(10)
    def _check_collision(self, obj, is_box=False):
        is_collision = False
        name = obj.get_name()
        self.logger.debug(f"start to check {name} collision")
        for c_obj in self.collision_objects + self.distractor + self.scene_boxes:
            if is_box:
                if c_obj in self.collision_objects:
                    continue
            
            if c_obj.check_collision(obj):
                is_collision = True
                # print(c_obj.get_name())
                self.logger.debug(f"end to check {name} collision")
                return is_collision
        
        for f_obj in self.scene_furniture + self.scene_assembled:
            res = f_obj.respondable
            if res == obj:
                continue
            if res.check_collision(obj):
                is_collision = True
                # print(res.get_name())
                self.logger.debug(f"end to check {name} collision")
                return is_collision
        self.logger.debug(f"end to check {name} collision")
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

    def randomize_camera(self, view_type: CameraViewType, location: CameraLocation):
        """
        case1. view grasping workspace and camera top or side

        """
        self.logger.debug("start randomize camera pose")
        # randomize view point
        view_range = view_type.value
        view_point = list(np.random.uniform(view_range[0], view_range[1]))
        self.camera_manager.set_rotation_base_position(view_point, relative_to=self.world_frame)

        # randomize position
        camera_range = location.value
        random_position = list(np.random.uniform(camera_range[0], camera_range[1]))
        self.camera_manager.set_position(random_position, self.world_frame)
        
        # randomize perspective angle
        self.camera_manager.set_random_angle()
        
        self.pr.step()

        self.logger.debug("end randomize camera pose")

    #endregion
    def set_stable(self):
        for fn in self.scene_furniture + self.scene_assembled:
            fn.set_dynamic(True)
        self.pr.step()
        for i in range(10):
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

    def reset(self, assembly_num, furniture_num, table_texture_type, wood_texture_type):
        self.logger.info("start to reset scene")
        self.texture_manager.reset()
        self.camera_manager.reset()
        self.camera_manager.set_activate(False)

        for distractor in self.distractor:
            distractor.remove()
        self.distractor = []
        
        # if opposed_mode => camera location is opposed
        if np.random.rand() < 0.5:
            self._opposed_mode = True
        else:
            self._opposed_mode = False

        self.table_texture_type = table_texture_type
        self.wood_texture_type = wood_texture_type

        self.table_texture = self.texture_manager.get_randomize_texture(self.table_texture_type.value, refer=self._table_color)
        self.furniture_texture = self.texture_manager.get_randomize_texture(TextureType.gray_texture, refer=self._fn_color)
        self.wood_texture = self.texture_manager.get_randomize_texture(self.wood_texture_type.value, refer=self._fn_color)

        self.randomize_scene()
        self._randomize_distractor(5, is_table=True)
        self._randomize_distractor(5, is_table=False)
        self._randomize_boxes()

        self._sample_furnitures(assembly_num, furniture_num)
        self._randomize_furniture()
               
        
        self.set_stable()
        self.logger.info("End to reset scene")

    def step(self):
        """
        1. randomize furniture pose
        2. randomize camera pose
        3. (opt)randomize robot pose
        """
        fn = random.choice(self.scene_assembled + self.scene_furniture)
        self._randomize_furniture_pose(fn)
        self.logger.debug("start one step")
        self.set_stable()
        self.logger.debug("end one step")

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

    def move_robot_to_grasp_workspace(self, robot):
        
        random_position = list(np.random.uniform(self.grasp_range[0], self.grasp_range[1]))
        random_position += [0]
        
        temp_dummy = Dummy.create()
        temp_dummy.set_position(random_position, relative_to=self.grasp_workspace)
        target_position = temp_dummy.get_position()
        robot.move_to_target_pose(self.pr, target_position)
        
        temp_dummy.remove()

    def grasp_random_furniture(self):
        """
        move furniture to grasping state
        """
        robot = self.robot_manager.robots[2]
        self.move_robot_to_grasp_workspace(robot)
        # select grasping target furniture
        self.grasp_furniture = random.choice(self.scene_furniture)
        self.grasp_furniture.set_dynamic(False)
        self.pr.step()

        # move furniture to ee_tip for only possible case
        is_possible_grasp_point = False
        grasp_dummy = Dummy.create()
        self.logger.info("finding possible grasp case...")
        count = 0
        while not is_possible_grasp_point and count < 10:
            self.grasp_furniture.set_parent(None)
            grasp_pose = self.grasp_furniture.get_random_grasp_pose()
            grasp_dummy.set_pose(grasp_pose)
            self.grasp_furniture.set_parent(grasp_dummy)
            ee_pose = robot.get_ee_pose()
            grasp_dummy.set_pose(ee_pose)
            self.pr.step()
            is_possible_grasp_point = self._check_collision(self.grasp_furniture.respondable)
            count += 1
        if not count < 10:
            self.logger.info("fail to find possible case")
            grasp_dummy.remove()
            self.grasp_furniture.set_respondable(False)
            self.grasp_furniture.reset()
            return False
        else:
            self.logger.info("success to find poosible case")            
            self.grasp_furniture.set_respondable(True)
        
            grasp_state = False
            while not grasp_state:
                grasp_state = robot.grasp()
                self.pr.step()
            
            self.grasp_furniture.set_parent(robot.ee_tip)
            robot.set_grasp_base_pose()
            grasp_dummy.remove()

            return True

    def randomize_robot_ee_pose(self):
        robot = self.robot_manager.robots[2]
        robot.randomize_ee_pose(self.pr)

    def release_furniture(self):
        self.grasp_furniture.set_parent(None)
        self.grasp_furniture.set_respondable(False)
        self.grasp_furniture.reset()

