#!/usr/bin/env python3
# coding: utf-8

import os
from PIL import Image
from pathlib import Path
import sys


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

    def dump_for_learn(self, size, test_span, class_num):
        dir_num = len(self._img_dirs)
        one_class_num = dir_num // class_num
        for k in range(class_num):
            one_class_dirs = self._img_dirs[k * one_class_num : (k + 1) * one_class_num]
            for i, img_dir in enumerate(one_class_dirs):

                img_paths = [str(p) for p in Path(img_dir).glob("./000.jpg")]
                img_paths.sort()
                if i % test_span == 0:
                    dir_names = [self._all_dir, self._test_dir]
                else:
                    dir_names = [self._all_dir, self._train_dir]
                output_dirs = [
                    temp + "{:02d}/{:03d}".format(k, i) for temp in dir_names
                ]
                for output_dir in output_dirs:
                    os.makedirs(output_dir)
                for j, img_path in enumerate(img_paths):
                    print(img_path)
                    image = Image.open(img_path)
                    image = image.resize(size)
                    for output_dir in output_dirs:
                        image.save(output_dir + "/{:03d}.jpg".format(j))

    def dump_finish_same(self, size, test_span, class_num, finish_count):
        dir_num = len(self._img_dirs)
        one_class_num = dir_num // class_num
        for k in range(class_num):
            one_class_dirs = self._img_dirs[k * one_class_num : (k + 1) * one_class_num]
            for i, img_dir in enumerate(one_class_dirs):
                print(img_dir)
                img_paths = [str(p) for p in Path(img_dir).glob("./*")]
                img_paths.sort()
                if i % test_span == 0:
                    dir_names = [self._all_dir, self._test_dir]
                else:
                    dir_names = [self._all_dir, self._train_dir]
                output_dirs = [
                    temp + "{:02d}/{:03d}".format(k, i) for temp in dir_names
                ]
                for output_dir in output_dirs:
                    os.makedirs(output_dir)
                for j, img_path in enumerate(img_paths):
                    image = Image.open(img_path)
                    image = image.resize(size)
                    for output_dir in output_dirs:
                        image.save(output_dir + "/{:03d}.jpg".format(j))
                image = Image.open(img_paths[0])
                image = image.resize(size)
                image.save(output_dirs[0] + "/{:03d}.jpg".format(finish_count))

    def dump_resize_normal(self, size, test_span, step_num):
        for i, img_dir in enumerate(self._img_dirs):
            img_paths = [str(p) for p in Path(img_dir).glob("./*")]
            img_paths.sort()
            if i % test_span == 0:
                dir_names = self._test_dir
            else:
                dir_names = self._train_dir
            output_dir = dir_names + "{:03d}".format(i)
            os.makedirs(output_dir)

            img_num = len(img_paths)
            for i in range(step_num):
                if i < img_num:
                    image = Image.open(img_paths[i])
                    image = image.resize(size)
                image.save(output_dir + "/{:03d}.jpg".format(i))


if __name__ == "__main__":
    # argnum = len(sys.argv)

    # if argnum == 2:
    #     _, name = sys.argv
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMG_DIR = (
        "/home/assimilation/TAKUMI_SHIMIZU/wiping_ws/src/wiping/data/0106/image_raw/"
    )
    # OUTPUT_DIR = DATA_DIR + "image_compressed/"
    OUTPUT_DIR = CURRENT_DIR + "/../../NN/data/CAE/0106/cs0maker/"
    process = ImgPreprocess(IMG_DIR, OUTPUT_DIR)
    # process.extract(50, 200)
    process.dump_for_learn(
        size=(128 + 5, 96 + 5),
        test_span=4,
        class_num=4,
    )
    # process.dump_resize_normal(size=(128, 96), test_span=4, step_num=185)
