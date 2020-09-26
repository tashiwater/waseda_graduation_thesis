#!/usr/bin/env python3
# coding: utf-8
import os
from pathlib import Path
import shutil


def rename(input_dir, output_dir, use_span, save_span, start_num):
    folder_names = ["image_raw/", "motion_yaml/", "tactile_raw/"]
    extensions = ["", ".yaml", ".csv"]
    for folder_name, extension in zip(folder_names, extensions):
        file_paths = [str(p) for p in Path(input_dir + folder_name).glob("./*")]
        file_paths.sort()
        num = start_num
        out = output_dir + folder_name
        # os.makedirs(out)
        for i, file_path in enumerate(file_paths):
            index = i % use_span
            use = False
            if index == 0:
                #     output_name = "{}/{}{}".format(out, num, extension)
                #     num += 1
                #     use = True
                # if index == 1:
                output_name = "{}/{}{}".format(out, num, extension)
                num += save_span
                use = True
            if use:
                if folder_name == "image_raw/":
                    shutil.copytree(file_path, output_name)
                else:
                    shutil.copy(file_path, output_name)


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = "/home/assimilation/TAKUMI_SHIMIZU/wiping/data/0926/"
    OUTPUT_DIR = CURRENT_DIR + "/../data/renamed/"
    rename(INPUT_DIR, OUTPUT_DIR, use_span=1, save_span=4, start_num=3)
