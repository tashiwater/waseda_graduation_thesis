#!/usr/bin/env python3
# coding: utf-8

import os
from PIL import Image
from pathlib import Path


class ImgPreprocess:
    def __init__(self, input_dir, output_dir):
        self._train_dir = output_dir + "train/"
        self._test_dir = output_dir + "test/"
        self._all_dir = output_dir + "all/"
        self._output_dir = output_dir
        self._img_dirs = [str(p) for p in Path(input_dir).glob("./*")]
        self._img_dirs.sort()

    def extract(self, min_index, max_index):
        for i, img_dir in enumerate(self._img_dirs):
            img_paths = [str(p) for p in Path(img_dir).glob("./*")]
            img_paths.sort()
            output_dir = self._output_dir + "{:03d}".format(i)
            os.makedirs(output_dir)
            for j, img_path in enumerate(img_paths[min_index:max_index]):
                image = Image.open(img_path)
                image.save(output_dir + "/{:03d}.jpg".format(j + min_index))

    def dump_for_learn(self, size, test_span):
        ### [TODO] this madgic number is class num
        class_num = 6
        dir_num = len(self._img_dirs)
        one_class_num = dir_num // class_num
        for k in range(6):
            one_class_dirs = self._img_dirs[k * one_class_num : (k + 1) * one_class_num]
            for i, img_dir in enumerate(one_class_dirs):
                img_paths = [str(p) for p in Path(img_dir).glob("./*")]
                img_paths.sort()
                if i % test_span == 0:
                    dir_names = [self._all_dir, self._test_dir]
                else:
                    dir_names = [self._all_dir, self._train_dir]
                output_dirs = [temp + "{}/{:03d}".format(k, i) for temp in dir_names]
                for output_dir in output_dirs:
                    os.makedirs(output_dir)
                for j, img_path in enumerate(img_paths):
                    image = Image.open(img_path)
                    image = image.resize(size)
                    for output_dir in output_dirs:
                        image.save(output_dir + "/{:03d}.jpg".format(j))

    def dump_resize_normal(self, size):
        for i, img_dir in enumerate(self._img_dirs):
            img_paths = [str(p) for p in Path(img_dir).glob("./*")]
            img_paths.sort()
            output_dir = self._output_dir + "{:03d}".format(i)
            os.mkdir(output_dir)
            for j, img_path in enumerate(img_paths):
                image = Image.open(img_path)
                image = image.resize(size)
                image.save(output_dir + "/{:03d}.jpg".format(j))


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../data/"
    IMG_DIR = DATA_DIR + "renamed/image_raw/"
    # OUTPUT_DIR = DATA_DIR + "image_compressed/"
    OUTPUT_DIR = "/home/assimilation/TAKUMI_SHIMIZU/waseda_graduation_thesis/CAE/data/"
    process = ImgPreprocess(IMG_DIR, OUTPUT_DIR)
    # process.extract(50, 200)
    process.dump_for_learn(size=(128 + 5, 96 + 5), test_span=4)
