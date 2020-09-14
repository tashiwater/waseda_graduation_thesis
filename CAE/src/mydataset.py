#!/usr/bin/env python3
# coding: utf-8
import torch
import torchvision
from pathlib import Path
from PIL import Image
import math
import torch.nn.functional as F
import numpy as np


class MyDataSetForCAE(torch.utils.data.Dataset):
    def __init__(self, dir_path, jpg_path="./*/*/*.jpg", noise=0):
        super().__init__()
        # self._noise = noise
        self._image_paths = [str(p) for p in Path(dir_path).glob(jpg_path)]
        self._image_paths.sort()

        self._imgs = [self.get_img(str(p)) for p in self._image_paths]
        self._len = len(self._imgs)
        self._noise = noise

    def __len__(self):
        return self._len

    def get_img(self, path):
        image = Image.open(path)
        image = image.convert("RGB")
        # image = image.transform(
        #     image.size, Image.AFFINE, (1, 0, 15, 0, 1, 15), Image.BILINEAR
        # )
        # image2 = np.array(image)

        image = image.resize((128, 96))
        image = torchvision.transforms.ToTensor()(image)
        return image

    def __getitem__(self, index):
        image = self._imgs[index]
        return image + torch.randn(image.shape) * self._noise, image

    @classmethod
    def save_img(self, tensor, filename):
        # hsv = torchvision.transforms.functional.to_pil_image(tensor, "HSV")
        # rgb = hsv.convert("RGB")
        # rgb.save(filename)
        torchvision.utils.save_image(tensor, filename)


class MyDataSetForAttention(torch.utils.data.Dataset):
    def __init__(self, dir_path, jpg_path="./*/*/*.jpg", noise=0):
        super().__init__()
        # self._noise = noise
        class_dirs = [str(p) for p in Path(dir_path).glob("./*")]
        class_dirs.sort()
        self._image_paths = []
        self._class_id = []
        class_id = 0
        for class_dir in class_dirs:
            img_paths = [str(p) for p in Path(class_dir).glob("./*/*.jpg")]
            img_paths.sort()
            self._class_id += [class_id for _ in range(len(img_paths))]
            class_id += 1
            self._image_paths += img_paths
        self._imgs = [self.get_img(str(p)) for p in self._image_paths]
        self._len = len(self._imgs)
        self._noise = noise

    def __len__(self):
        return self._len

    def get_img(self, path):
        image = Image.open(path)
        image = image.convert("RGB")
        # image = image.transform(
        #     image.size, Image.AFFINE, (1, 0, 15, 0, 1, 15), Image.BILINEAR
        # )
        # image2 = np.array(image)

        image = image.resize((128, 96))
        image = torchvision.transforms.ToTensor()(image)
        return image

    def __getitem__(self, index):
        image = self._imgs[index]
        return (
            image + torch.randn(image.shape) * self._noise,
            [image, self._class_id[index]],
        )

    @classmethod
    def save_img(self, tensor, filename):
        # hsv = torchvision.transforms.functional.to_pil_image(tensor, "HSV")
        # rgb = hsv.convert("RGB")
        # rgb.save(filename)
        torchvision.utils.save_image(tensor, filename)
