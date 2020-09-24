#!/usr/bin/env python3
# coding: utf-8
import torch
import torchvision
from pathlib import Path
from PIL import Image
import math
import torch.nn.functional as F
import numpy as np

import random
from .utils.argments.resize import resize_image
from .utils.argments.crop import random_crop_image
from .utils.argments.flip import random_flip_image
from .utils.argments.distort import random_distort


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
        super(MyDataSetForAttention, self).__init__()
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

        image = image.resize((128 + self.dsize, 96 + self.dsize))
        return image

    def transform(self, img):
        img = random_crop_image(img, self.size, test=self.test)
        if self.distort:
            img_distort = np.copy(img)
            if random.randrange(2):
                img_distort = random_distort(img_distort)
            img_distort = img_distort.astype(np.float32).transpose(2, 0, 1)
            img_distort /= 255.0
        else:
            img_distort = np.copy(img)
            img_distort = img_distort.astype(np.float32).transpose(2, 0, 1)
            img_distort /= 255.0
        img = img.astype(np.float32).transpose(2, 0, 1)
        img /= 255.0  # <- very important !!! (VAE)
        # if self.test:
        #    save_img(img, os.path.join("/home/assimilation","test_"+imgpath.split("/")[-1]))
        # else:
        #    save_img(img, os.path.join("/home/assimilation",imgpath.split("/")[-1]))
        # save_img(img, os.path.join("/home/assimilation","test2_"+imgpath.split("/")[-1]))
        # sys.exit()
        img_distort = torch.from_numpy(img_distort)
        img = torch.from_numpy(img)
        return img_distort, img

    def __getitem__(self, index):
        img = self._imgs[index]
        img_distort, img = self.transform(img)
        return (
            img_distort,
            [img, self._class_id[index]],
        )

    @classmethod
    def save_img(self, tensor, filename):
        # hsv = torchvision.transforms.functional.to_pil_image(tensor, "HSV")
        # rgb = hsv.convert("RGB")
        # rgb.save(filename)
        torchvision.utils.save_image(tensor, filename)
