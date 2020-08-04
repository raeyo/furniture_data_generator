from tqdm import tqdm
from PIL import Image, ImageChops
import os
import glob
import numpy as np
import pickle
import shutil

# remove image w/o connector or furniture parts
dataset_root = "/SSD1/joo/Dataset/furniture/SIM_dataset_v13"

folders = ["seg"]
error_buff = {}
for i in range(len(folders)):
    files = glob.glob(dataset_root + "/" + folders[i] + "/*")
    print(folders[i], len(files))
    error_buff[folders[i]] = 0
    for f in tqdm(files):
        img = Image.open(f)
        img_arr = np.array(img)

        mask_furn = img_arr[:, :, 1]
        obj_ids_furn = np.unique(mask_furn)
        img_f = Image.fromarray(mask_furn)

        if 0 in obj_ids_furn:
            obj_ids_furn = np.setdiff1d(obj_ids_furn, [0])

        masks = mask_furn == obj_ids_furn[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids_furn)
        temp_obj_ids = []
        temp_masks = []
        boxes = []

        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if int(xmax-xmin) < 1 or int(ymax-ymin) < 1 :
                continue
            temp_masks.append(masks[i])
            temp_obj_ids.append(obj_ids_furn[i])
            boxes.append([xmin, ymin, xmax, ymax])

        if len(boxes) == 0 or len(obj_ids_furn) == 0 or not img.getbbox() or not ImageChops.invert(img).getbbox():
            print("remove ! w/o furniture part ", f)
            error_buff[folders[i]] += 1
            # os.system("rm {}".format(f))

print(error_buff)
print(len(error_buff))
