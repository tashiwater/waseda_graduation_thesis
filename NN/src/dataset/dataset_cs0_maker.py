#!/usr/bin/env python3
# coding: utf-8
import torch
import torchvision
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

import random
from .CAE_utils.argments.crop import random_crop_image
from .CAE_utils.argments.distort import color_argumentation


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, dir_path, img_size=(128, 96), is_test=False, dsize=5, noise=0):
        super(MyDataSet, self).__init__()

        self.size = img_size
        self.test = is_test
        self.distort = not is_test
        self.dsize = dsize
        self._noise = noise
        class_dirs = [str(p) for p in Path(dir_path).glob("./*")]
        class_dirs.sort()
        self._image_paths = []
        self._class_id = []
        class_id = 0
        for class_dir in class_dirs:
            img_paths = [str(p) for p in Path(class_dir).glob("./*/*.jpg")]
            """
            ### add for trans learning
            files = [str(p) for p in Path(class_dir).glob("./*")]
            files.sort()
            img_paths = []
            for p in files:
                img_path = [str(p) for p in Path(p).glob("./*.jpg")][0]
                img_paths.append(img_path)
            ###
            """
            img_paths.sort()
            self._class_id += [class_id for _ in range(len(img_paths))]
            class_id += 1
            self._image_paths += img_paths
        self._imgs = [self.get_img(str(p)) for p in self._image_paths]
        # self._imgs = [self.get_img(str(p)) for p in self._image_paths] * 40
        # self._class_id = self._class_id * 40
        self._len = len(self._imgs)
        cs0_path = dir_path + "/cs0.csv"
        self.cs0_df = np.loadtxt(cs0_path, delimiter=",")
        self.cs0_df = torch.tensor(self.cs0_df.astype(np.float32))

    def __len__(self):
        return self._len

    def get_img(self, path):
        image = Image.open(path)
        image = image.convert("RGB")
        image = image.resize((self.size[0] + self.dsize, self.size[1] + self.dsize))
        # image = image.resize((128, 96))
        # image = torchvision.transforms.ToTensor()(image)
        return image

    def transform(self, img):
        img = random_crop_image(img, self.size, test=self.test)

        img_distort = img.copy()
        if self.distort:
            if random.randrange(2):  # add random at twice
                img_distort = color_argumentation(img_distort, pil_range=[0.8, 1.2])
        img_distort = torchvision.transforms.ToTensor()(img_distort)
        img = torchvision.transforms.ToTensor()(img)
        return img_distort, img

    def __getitem__(self, index):
        img = self._imgs[index]
        img_distort, img = self.transform(img)
        return (
            img_distort + torch.randn(img_distort.shape) * self._noise,
            self.cs0_df[index],
        )

    @classmethod
    def save_img(self, tensor, filename):
        # hsv = torchvision.transforms.functional.to_pil_image(tensor, "HSV")
        # rgb = hsv.convert("RGB")
        # rgb.save(filename)
        torchvision.utils.save_image(tensor, filename)


class OneDataSet(MyDataSet):
    def __init__(self, dir_path, img_size=(128, 96), is_test=False, dsize=5, noise=0):

        self.size = img_size
        self.test = is_test
        self.distort = not is_test
        self.dsize = dsize
        self._noise = noise
        self._image_paths = []
        class_id = 0
        img_paths = [str(p) for p in Path(dir_path).glob("./*.jpg")]
        img_paths.sort()
        self._class_id = [class_id for _ in range(len(img_paths))]
        self._image_paths += img_paths
        self._imgs = [self.get_img(str(p)) for p in self._image_paths]
        self._len = len(self._imgs)
