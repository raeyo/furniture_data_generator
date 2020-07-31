from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.const import ObjectType

import numpy as np

class MyPanda(object):
    def __init__(self, instance=0):
        self.arm = Panda(instance)
        self.gripper = PandaGripper(instance)
        self.initial_position = self.arm.get_position()
        self.initial_joint_positions = self.arm.get_joint_positions()
        self.visible_parts = self._extract_visible_parts()
        self.respondable_parts = self._extract_respondable_parts()

    def get_random_pose(self):
        dif = np.ones(7)*0.5
        return list(np.random.uniform(self.initial_joint_positions - dif,
                                                         self.initial_joint_positions + dif))

    def _extract_visible_parts(self):
        visible_parts = []
        visible_arm = self.arm.get_visuals()
        visible_grip = self.gripper.get_visuals()
        for v_a in visible_arm:
            visible_parts += v_a.ungroup()
        for v_g in visible_grip:
            visible_parts += v_g.ungroup()

        return visible_parts
    
    def _extract_respondable_parts(self):
        tree = self.arm.get_objects_in_tree(ObjectType.SHAPE, exclude_base=False)
        res_arm = [obj for obj in tree if 'respondable' in obj.get_name()]
        tree = self.gripper.get_objects_in_tree(ObjectType.SHAPE, exclude_base=False)
        res_grip = [obj for obj in tree if 'respondable' in obj.get_name()]

        return res_arm + res_grip

    def get_random_xy_position(self):
        dif = np.ones(3)*0.05
        dif[2] = 0
        return list(np.random.uniform(self.initial_position - dif,
                                      self.initial_position + dif))

    def set_random_pose(self):
        self.arm.set_position(self.get_random_xy_position())
        self.arm.set_joint_positions(self.get_random_pose())

    def get_visible_objects(self):
        
        return self.visible_parts

    def get_respondable_objects(self):
        return self.respondable_parts
