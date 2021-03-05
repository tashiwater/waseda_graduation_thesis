#!/usr/bin/env python3
# coding: utf-8

import os
from pathlib import Path
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
mov_dirs = [str(p) for p in Path(DATA_DIR + "input/").glob("./*")]
mov_dirs.sort()
for mov_dir in mov_dirs:
    cap = cv2.VideoCapture(mov_dir)
    if not cap.isOpened():
        continue
    basename_without_ext = os.path.splitext(os.path.basename(mov_dir))[0]
    dir_path = DATA_DIR + "result/{}".format(basename_without_ext)
    os.makedirs(dir_path, exist_ok=True)
    count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            count += 1
            if count % 100 != 0:
                continue
            x, y = 500, 0
            w, h = 450, 900
            img_trim = frame[y : y + h, x : x + w]

            img_trim = cv2.resize(img_trim, None, fx=1 / 4, fy=1 / 4)
            cv2.imwrite("{}/{:03d}.jpg".format(dir_path, count), img_trim)

        else:
            break
    cap.release()