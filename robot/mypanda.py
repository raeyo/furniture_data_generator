from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.const import ObjectType
from pyrep.errors import ConfigurationPathError

import numpy as np


class MyPanda(object):
    def __init__(self, instance=0, is_gripper=True):
        self.arm = Panda(instance)
        self.is_gripper = is_gripper
        if self.is_gripper:
            self.gripper = PandaGripper(instance)
        self.ee_tip = self.arm.get_tip()
        self.initial_position = self.arm.get_position()
        self.initial_joint_positions = self.arm.get_joint_positions()
        self.visible_parts = self._extract_visible_parts()
        self.respondable_parts = self._extract_respondable_parts()
        print("instance:", instance, "--"*10)
        for p in self.respondable_parts:
            print(p.get_name())

        self.workspace = self.get_workspace()

    def get_workspace(self):
        ee_position = self.ee_tip.get_position()
        ee_position = np.array(ee_position)
        plus_delta = np.array([0.1, 0.2, 0.05])
        minus_delta = np.array([0.1, 0.2, 0.3])
        pos_min, pos_max = ee_position - minus_delta, ee_position + plus_delta
        workspace = [(pos_min), (pos_max)]

        return workspace

    def get_random_ee_pose(self):
        return np.random.uniform(self.workspace[0], self.workspace[1])

    def get_random_pose(self, relative=False):
        dif = np.ones(7)*0.5
        if not relative:
            return list(np.random.uniform(self.initial_joint_positions - dif,
                                          self.initial_joint_positions + dif))
        else:
            current_joint_positions = self.arm.get_joint_positions()
            return list(np.random.uniform(current_joint_positions - dif,
                                          current_joint_positions + dif))

    def _extract_visible_parts(self):
        visible_parts = []
        visible_arm = self.arm.get_visuals()
        for v_a in visible_arm:
            visible_parts += v_a.ungroup()
        if self.is_gripper:
            visible_grip = self.gripper.get_visuals()
            for v_g in visible_grip:
                visible_parts += v_g.ungroup()

        return visible_parts
    
    def _extract_respondable_parts(self):
        tree = self.arm.get_objects_in_tree(ObjectType.SHAPE, exclude_base=False)
        res = []
        res += [obj for obj in tree if 'respondable' in obj.get_name()]
        if self.is_gripper:
            tree = self.gripper.get_objects_in_tree(ObjectType.SHAPE, exclude_base=False)
            res += [obj for obj in tree if 'respondable' in obj.get_name()]
            
        return res

    def get_random_xy_position(self):
        dif = np.ones(3)*0.05
        dif[2] = 0
        return list(np.random.uniform(self.initial_position - dif,
                                      self.initial_position + dif))

    def set_random_pose(self, relative=False):
        self.arm.set_position(self.get_random_xy_position())
        self.arm.set_joint_positions(self.get_random_pose(relative))

    def get_visible_objects(self):
        
        return self.visible_parts

    def get_respondable_objects(self):
        return self.respondable_parts

    def release(self):
        return self.gripper.actuate(1, 0.1)
    
    def grasp(self):
        return self.gripper.actuate(0, 0.1)
    
    def get_ee_pose(self):
        return self.ee_tip.get_pose()

    def _get_waypoints(self, target_position: list, target_orientation: list, num_pts: int):
        if num_pts < 2:
            return [target_position], [target_orientation]

        cur_pos = self.ee_tip.get_position()
        cur_ori = self.ee_tip.get_orientation()
        
        waypoints = np.linspace(cur_pos, target_position, num_pts)
        wayoris = np.linspace(cur_ori, target_orientation, num_pts)

        return waypoints, wayoris

    def _move_to_target_pose(self, pr, target_position, target_orientation):
        try:
            path = self.arm.get_path(position=target_position, 
                                     euler=target_orientation,
                                     ignore_collisions=True)
        except ConfigurationPathError as e:
            return False
        
        done = False
        while not done:
            done = path.step()
            pr.step()
        
        return True
            
    def move_to_target_pose(self, pr, target_position, target_orientation=[0, np.pi, 0]):
        reach_state = False
        count = 0
        num_pts = 1
        move_step = 0
        while not reach_state and count < 5:
            num_pts = (num_pts + 1) - move_step
            way_pts, way_oris = self._get_waypoints(target_position, target_orientation, num_pts)
            move_step = 0
            for pt, ori in zip(way_pts, way_oris):
                if not self._move_to_target_pose(pr, pt, ori):
                    print("Can not find")
                    reach_state = False
                    break
                else:
                    reach_state = True
                    move_step += 1
            if not reach_state:
                count += 1

    def randomize_ee_pose(self, pr):
        target_position = list(np.array(self.grasp_base_position) + (np.random.rand(3) - 0.5) / 10)
        target_position[2] += 0.05 
        target_orientation = list(np.array(self.grasp_base_orientation) + (np.random.rand(3) - 0.5) / 10)
        self.move_to_target_pose(pr, target_position=target_position, target_orientation=target_orientation)

    def reset(self):
        self.arm.set_joint_positions(self.initial_joint_positions)

    def set_grasp_base_pose(self):
        self.grasp_base_position = self.ee_tip.get_position()
        self.grasp_base_orientation = self.ee_tip.get_orientation()
