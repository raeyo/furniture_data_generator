import os
import time


print(time.strftime("%m.%d. %H:%M:%S", time.localtime(time.time())))
# count the number of dataset
data_root = '/SSD1/joo/Dataset/furniture/SIM_dataset_v13'

dir_list = os.listdir(data_root)

for dir_name in dir_list:
    file_list = os.listdir(os.path.join(data_root, dir_name))
    print("[{}] {}".format(dir_name, len(file_list)))
    