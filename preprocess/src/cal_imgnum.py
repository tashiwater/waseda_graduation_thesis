#!/usr/bin/env python3
# coding: utf-8

import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
IMG_DIR = "/home/assimilation/TAKUMI_SHIMIZU/wiping/data/1210/image_raw/"
img_dirs = [str(p) for p in Path(IMG_DIR).glob("./*")]
img_dirs.sort()
img_nums = []
for i, img_dir in enumerate(img_dirs):
    img_paths = [str(p) for p in Path(img_dir).glob("./*")]
    img_nums.append(len(img_paths))
max_num = max(img_nums)
index = img_nums.index(max_num)
print(index, max_num)
plt.hist(img_nums, bins=len(img_nums))
plt.show()
