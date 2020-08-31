#!/usr/bin/env python3
# coding: utf-8

import os
from PIL import Image
from pathlib import Path


class ImgPreprocess:
    def __init__(self, input_dir, output_dir):
        self._output_dir = output_dir
        self._img_dirs = [str(p) for p in Path(input_dir).glob("./*")]
        self._img_dirs.sort()

    def extract(self, min_index, max_index):
        for i, img_dir in enumerate(self._img_dirs):
            img_paths = [str(p) for p in Path(img_dir).glob("./*")]
            img_paths.sort()
            output_dir = self._output_dir + "{:03d}".format(i)
            os.mkdir(output_dir)
            for j, img_path in enumerate(img_paths[min_index:max_index]):
                image = Image.open(img_path)
                image.save(output_dir + "/{:03d}.jpg".format(j + min_index))

    def resize(self):
        for i, img_dir in enumerate(self._img_dirs):
            img_paths = [str(p) for p in Path(img_dir).glob("./*")]
            img_paths.sort()
            output_dir = self._output_dir + "{:03d}".format(i)
            os.mkdir(output_dir)
            for j, img_path in enumerate(img_paths):
                image = Image.open(img_path)
                image = image.resize((128, 96))
                image.save(output_dir + "/{:03d}.jpg".format(j))


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../data/"
    IMG_DIR = DATA_DIR + "image_compressed/"
    OUTPUT_DIR = DATA_DIR + "image_extract/"
    process = ImgPreprocess(IMG_DIR, OUTPUT_DIR)
    process.extract(50, 200)
    # process.resize()
