import os
import glob
import numpy as np
from tqdm import tqdm


dataset_root = "/SSD1/joo/Dataset/furniture/SIM_dataset_v12"
folders = ["seg", "hole_mask", "rgb", "depth_value"]


print("-----" * 20)
print("All files!")
all_files = []
for i in range(len(folders)):
    files = [file_name[:-4] for file_name in os.listdir(os.path.join(dataset_root, folders[i]))]
    all_files.append(files)
    print(folders[i], len(files))


print("-----" * 20)
print("Intersect !")
for i in range(0, len(all_files)-1):
    if i == 0:
        intersect = list(set(all_files[i]) & set(all_files[i+1]))
        print(folders[i], len(intersect))
    else:
        intersect = list(set(intersect) & set(all_files[i+1]))
    print(folders[i+1], len(intersect))


print("-----" * 20)
print("Intersect !")
for i, files in enumerate(all_files):
    diff = list(set(files) - set(intersect))
    print(folders[i], len(diff))
    for file_name in tqdm(diff):
        suffix = '.png' if folders[i] != 'depth_value' else '.npy'
        f = os.path.join(dataset_root, folders[i], file_name + suffix)
        os.system("rm {}".format(f))
