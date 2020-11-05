#!/usr/bin/env python3
# coding: utf-8
import torch
import torchvision
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import random
from .CAE_utils.argments.crop import random_crop_image
from .CAE_utils.argments.distort import color_argumentation


class MyDataSet(torch.utils.data.Dataset):
    def __init__(
        self, csv_path, img_path, img_size=(128, 96), is_test=False, dsize=5, noise=0
    ):
        super(MyDataSet, self).__init__()
        self._paths = [str(p) for p in Path(csv_path).glob("./*.csv")]
        self._paths.sort()

        self._datas = []
        for path in self._paths:
            df = pd.read_csv(path)
            x = torch.tensor(df.values.astype(np.float32))
            self._datas.append(x)
        self._len = len(self._datas)

        self.size = img_size
        self.test = is_test
        self.distort = not is_test
        self.dsize = dsize
        self._noise = noise
        class_dirs = [str(p) for p in Path(img_path).glob("./*")]
        class_dirs.sort()
        self._imgs = []
        for class_dir in class_dirs:
            img_paths = [str(p) for p in Path(class_dir).glob("./*.jpg")]
            img_paths.sort()
            imgs = [self.get_img(str(p)) for p in img_paths]
            imgs = torch.stack(imgs)
            self._imgs.append(imgs)
        self._imgs = torch.stack(self._imgs)
        self._img_shape = self._imgs[0][0].shape

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        # print(self._datas[index][1:].shape)
        return (
            (
                self._datas[index][0:-1],
                self._imgs[index][0:-1] + torch.randn(self._img_shape) * self._noise,
            ),
            (self._datas[index][1:], self._imgs[index][1:]),
        )

    def get_img(self, path):
        image = Image.open(path)
        image = image.convert("RGB")
        # image = image.resize((self.size[0] + self.dsize, self.size[1] + self.dsize))
        image = image.resize((128, 96))
        image = torchvision.transforms.ToTensor()(image)
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

    @classmethod
    def save_img(self, tensor, filename):
        # hsv = torchvision.transforms.functional.to_pil_image(tensor, "HSV")
        # rgb = hsv.convert("RGB")
        # rgb.save(filename)
        torchvision.utils.save_image(tensor, filename)
