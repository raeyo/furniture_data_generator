from pyrep.objects.shape import Shape
from pyrep.const import ObjectType
from pyrep.objects.dummy import Dummy
from externApi.fileApi import *
from const import *

import numpy as np
import random

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

        self.obj_frame = Dummy(furniture_name + "_obj_frame")
        self.grasp_points = load_yaml_to_dic(join("./grasps", furniture_name + "_grasp.yaml"))["grasp_points"]
        # self.create_grasp_dummys()

        # self._initialize_bbox()

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
        
    def create_grasp_dummys(self):
        grasp_base = Dummy.create()
        grasp_base.set_name(self.name + "_grasp_base")
        if len(self.grasp_points) > 100:
            candidate = random.sample(self.grasp_points, 100)
        else:
            candidate = self.grasp_points
        for grasp_point in candidate:
            pose = grasp_point[0] + grasp_point[1]
            grasp_dummy = Dummy.create()
            grasp_dummy.set_pose(pose, relative_to=grasp_base)
            grasp_dummy.set_parent(grasp_base)
        grasp_base.set_parent(self.obj_frame)
        grasp_base.set_pose([0, 0, 0, 0, 0, 0, 1], relative_to=self.obj_frame)

    def get_random_grasp_pose(self):
        grasp_point = random.choice(self.grasp_points)
        pose = grasp_point[0] + grasp_point[1]
        temp_dummy = Dummy.create()
        temp_dummy.set_pose(pose, relative_to=self.obj_frame)
        grasp_pose = temp_dummy.get_pose()
        temp_dummy.remove()

        return grasp_pose
        
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
        self.asm_num = len(self.sub_parts)

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
        # pose는 0000000 이면 안됨
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