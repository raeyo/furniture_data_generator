import os
from datetime import datetime
import numpy as np
from PIL import Image
from skimage import io
from os.path import join
import time
import sys
import argparse
import pickle
import logging

from environment import DREnv
from const import *
from externApi.fileApi import *
import random
import warnings
warnings.filterwarnings('error')

class Generator(object):

    def __init__(self, scene, max_img_num, max_fn_num, 
                 logger,
                 ep_length=5, 
                 headless=False, 
                 process_id=1,
                 save_root=None,
                 save_dir=None,
                 dataset_ver=0,
                 ):
        # object number setting
        self.fn_min = 0
        self.fn_max = max_fn_num
        self.furniture_num = None

        # initialize environment
        self.process_id = process_id
        self.env = DREnv(logger=logger,
                         headless=headless,
                         dataset_ver=dataset_ver,
                         process_id=process_id)
        self.dataset_ver = dataset_ver

        # data generate setting        
        self.img_max = max_img_num
        self.img_num = 0
        self.img_name = ""
        self.ep_length = ep_length
        self.ep_num = 0
        self.uint16_conversion = 10000
        """
        detph value
        simulation range 0.0001 ~ 10(m) => 0 ~ 1 
        raw_data * conversion(10000) => 0 ~ 10000(mm) == 0 ~ 10000
        conversion_data 1 mean == 1 mm
        """ 

        # data directory setting
        self.save_root = save_root
        self.save_dir = save_dir
        self._initialize_dir()

        # logger
        self.logger = logger

        # furniture num
        self.furniture_num = 0
        self.assembly_num = 0

    def _set_img_name(self):
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        self.img_name = "fn{}_asm{}_{}_and_{}_time{}.png".format(self.furniture_num,
                                                                 self.assembly_num,
                                                                 self.table_texture.name,
                                                                 self.wood_texture.name,
                                                                 timestamp)

    def _set_img_name_grasp(self):
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        self.img_name = "{}_{}_{}_and_{}_time{}.png".format(self.camera_location.name,
                                                            self.view_type.name,
                                                            self.table_texture.name,
                                                            self.wood_texture.name,
                                                            timestamp)

    def _is_ended(self):
        if self.img_num < self.img_max:
            return False
        else:
            return True
    
    def _initialize_dir(self):
        if self.save_root == None:
            self.save_root = os.path.dirname(os.path.realpath(__file__))
        if self.save_dir == None:
            self.save_dir = "image"

        self.save_path = join(self.save_root, self.save_dir)
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        # data save path
        self.rgb_dir = join(self.save_path, "rgb")
        check_and_create_dir(self.rgb_dir)
        self.depth_dir = join(self.save_path, "depth_value")
        check_and_create_dir(self.depth_dir)
        self.seg_dir = join(self.save_path, "seg")
        check_and_create_dir(self.seg_dir)
        self.hmask_dir = join(self.save_path, "hole_mask")
        check_and_create_dir(self.hmask_dir)
        self.asm_dir = join(self.save_path, "asm_mask")
        check_and_create_dir(self.asm_dir)

    def save_image(self):
        rgb_image, depth_raw, seg, hole_mask, asm_mask = self.env.get_images()

        self._set_img_name()
        np_name = self.img_name.replace(".png", ".npy") # save numpy 
        seg_new = np.uint8(seg * 255)
        rgb_image = Image.fromarray(np.uint8(rgb_image * 255))        
        depth_value = np.uint16(depth_raw * self.uint16_conversion)
        seg = Image.fromarray(np.uint8(seg * 255))
        hole_mask = Image.fromarray(hole_mask)
        asm_mask = Image.fromarray(asm_mask)

        self.logger.debug("start to save image")        
        try:
            rgb_image.save(join(self.rgb_dir, self.img_name))
            self.logger.debug("save rgb")
            np.save(join(self.depth_dir, np_name), depth_value)
            self.logger.debug("save_depth")
            seg.save(join(self.seg_dir, self.img_name))
            self.logger.debug("save seg")
            hole_mask.save(join(self.hmask_dir, self.img_name))
            asm_mask.save(join(self.asm_dir, self.img_name))
            self.logger.debug("save mask")      
        
        except FileNotFoundError:
            self.logger.error("No save directory")
    
    def save_image_grasp(self):
        rgb_image, depth_raw, seg, hole_mask, _ = self.env.get_images()

        self._set_img_name_grasp()
        np_name = self.img_name.replace(".png", ".npy") # save numpy 
        seg_new = np.uint8(seg * 255)
        rgb_image = Image.fromarray(np.uint8(rgb_image * 255))        
        depth_value = np.uint16(depth_raw * self.uint16_conversion)
        seg = Image.fromarray(np.uint8(seg * 255))
        hole_mask = Image.fromarray(hole_mask)

        self.logger.debug("start to save image")        
        check_and_create_dir(join(self.rgb_dir, self.furniture_name))
        try:
            rgb_image.save(join(self.rgb_dir, self.furniture_name, self.img_name))
            self.logger.debug("save rgb")
            np.save(join(self.depth_dir, np_name), depth_value)
            self.logger.debug("save_depth")
            seg.save(join(self.seg_dir, self.img_name))
            self.logger.debug("save seg")
            hole_mask.save(join(self.hmask_dir, self.img_name))
            self.logger.debug("save mask")
            
        except FileNotFoundError:
            self.logger.error("No save directory")

    def run_episode(self):
        """randomize part
        1. only one primitive part: 0.25
        2. only one assembly part: 0.25
        3. many primitive part: 0.25
        4. one assembly part + other parts: 0.25
        """
        random_value = np.random.rand()
        if random_value < 0.25:
            furniture_num = 1
            assembly_num = 0
        elif 0.25 < random_value < 0.5:
            furniture_num = 0
            assembly_num = 1
        elif 0.5 < random_value < 0.75:
            furniture_num = np.random.randint(2, 7) # 2, 3, 4, 5, 6 
            assembly_num = 0
        else:
            furniture_num = np.random.randint(1, 4) # 1, 2, 3 
            assembly_num = 1
        self.furniture_num = furniture_num
        self.assembly_num = assembly_num
        self.table_texture = random.choice(list(TableTextureType))
        self.wood_texture = random.choice(list(WoodTextureType))
        self.env.reset(assembly_num, furniture_num, self.table_texture, self.wood_texture)
        self.env.camera_manager.set_activate(True)
        for i in range(self.ep_length):
            self.env.step()
            
            for self.camera_location in CameraLocation:
                self.view_type = CameraViewType.workspace
                self.env.randomize_camera(view_type=self.view_type,
                                          location=self.camera_location)
                
                self.save_image()
                self.logger.info(f"successfully saved image: {self.img_name}")
                self.img_num += 1
        self.env.camera_manager.set_activate(False)

    def run_grasp_episode(self):
        """
        only primitive part
        
        """
        self.furniture_num = np.random.randint(1, 7)
        self.assembly_num = 0
        self.table_texture = random.choice(list(TableTextureType))
        self.wood_texture = random.choice(list(WoodTextureType))
        self.env.reset(self.assembly_num, self.furniture_num, self.table_texture, self.wood_texture)
        for i in range(self.ep_length):
            # 1. move robot to grasp pose
            # 2. pick furniture and grasp it
            while not self.env.grasp_random_furniture():
                continue
            self.furniture_name = self.env.grasp_furniture.name
            self.env.camera_manager.set_activate(True)
            for j in range(5):
                # 3. move
                self.env.randomize_robot_ee_pose()
                # case 1. left and see grasp space
                self.camera_location = CameraLocation.left
                self.view_type = CameraViewType.graspspace
                self.env.randomize_camera(view_type=self.view_type,
                                          location=self.camera_location)
                self.save_image_grasp()
                self.logger.info(f"successfully saved image: {self.img_name}")
                # case 2. top and see grasp space
                self.camera_location = CameraLocation.top
                self.view_type = CameraViewType.graspspace
                self.env.randomize_camera(view_type=self.view_type,
                                          location=self.camera_location)
                self.save_image_grasp()
                self.logger.info(f"successfully saved image: {self.img_name}")
                # case 3. top and see work space
                self.camera_location = CameraLocation.top
                self.view_type = CameraViewType.graspspace
                self.env.randomize_camera(view_type=self.view_type,
                                          location=self.camera_location)
                self.save_image_grasp()
                self.logger.info(f"successfully saved image: {self.img_name}")

                self.img_num += 1
            self.env.release_furniture()
            self.env.camera_manager.set_activate(False)

    def start(self):
        while not self._is_ended(): 
            if self.dataset_ver == 23:
                self.run_grasp_episode()
            else:
                self.run_episode()
            self.ep_num += 1            
        self.env.shutdown()
    
    def stop(self):
        self.env.shutdown()

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    # basic setting
    parser.add_argument("--pid", type=int, default=1, help="process id. 1 from 16")
    parser.add_argument("--scene", type=str, default="assembly_env_SNU", help="scene file")
    parser.add_argument("--headless", action='store_true', help='no gui if true')
    # data generate setting
    parser.add_argument("--max_fn_num", type=int, default=2, help="maximum furniture number per scene")
    parser.add_argument("--max_img_num", type=int, default=10000, help="maximum image number")
    parser.add_argument("--ep_length", type=int, default=5, help="number of episode")
    # parser.add_argument("--save_root", type=str, default="/SSD1/joo/Dataset/furniture", help="saving directory root")
    parser.add_argument("--save_root", type=str, default="/home/raeyo/data_set", help="saving directory root")
    
    parser.add_argument("--dataset_ver", type=int, default=24, help="saving directory")
    args = parser.parse_args()

    # logger
    logger = logging.getLogger(f"Generator{args.pid}")
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.setLevel(level=logging.INFO)
    # logger.setLevel(level=logging.DEBUG)
    

    save_dir = "SIM_dataset_v" + str(args.dataset_ver)
    # generate data
    generator = Generator(scene=args.scene,
                          max_img_num=args.max_img_num,
                          max_fn_num=args.max_fn_num,
                          logger=logger,
                          ep_length=args.ep_length,
                          save_root=args.save_root,
                          save_dir=save_dir,
                          dataset_ver=args.dataset_ver,
                          headless=args.headless,
                          process_id=args.pid,
                          ) 
    start_time1 = time.time()
    try:
        generator.start()
    except RuntimeWarning:
        print("[ERROR] stop simulation")
        generator.stop()
