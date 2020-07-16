#!/usr/bin/env python3
# coding: utf-8
import torch
import torchvision
from pathlib import Path
from PIL import Image
import math


class MyDataSetForCAE(torch.utils.data.Dataset):
    def __init__(self, dir_path):
        super().__init__()
        # self._noise = noise
        self._image_paths = [str(p) for p in Path(dir_path).glob("./*/*/*.jpg")]
        self._image_paths.sort()

        self._imgs = [self.get_img(str(p)) for p in self._image_paths]
        self._len = len(self._imgs)

    def __len__(self):
        return self._len

    def get_img(self, path):
        image = Image.open(path)
        image = image.convert("RGB")
        # image = image.transform(
        #     image.size, Image.AFFINE, (1, 0, 15, 0, 1, 15), Image.BILINEAR
        # )
        image = torchvision.transforms.ToTensor()(image)
        return image

    def __getitem__(self, index):
        image = self._imgs[index]
        return image, image
