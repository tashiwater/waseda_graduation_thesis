#!/usr/bin/env python3
# coding: utf-8

import os
from pathlib import Path
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
img_dirs = [str(p) for p in Path(DATA_DIR + "input/").glob("./*")]
img_dirs.sort()
for img_dir in img_dirs:

    img_paths = [str(p) for p in Path(img_dir).glob("./*.jpg")]
    img_paths.sort()
    if img_paths == []:
        continue
    print(img_dir)
    img_array = []
    for filename in img_paths:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    basename_without_ext = os.path.splitext(os.path.basename(img_dir))[0]
    name = DATA_DIR + "result/{}.mp4".format(basename_without_ext)
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"MP4V"), 10, size)

    for img in img_array:
        out.write(img)
    out.release()