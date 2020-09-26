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
from utils.argments.crop import random_crop_image
from utils.argments.distort import color_argumentation


class MyDataSetForCAE(torch.utils.data.Dataset):
    def __init__(self, dir_path, jpg_path="./*/*/*.jpg", noise=0):
        super(MyDataSetForCAE, self).__init__()
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


class MyDataSetForCAE2(torch.utils.data.Dataset):
    def __init__(self, dir_path, jpg_path="./*/*/*.jpg", dsize=5):
        super(MyDataSetForCAE2, self).__init__()

        self.dsize = dsize
        self.size = (128, 96)
        self._image_paths = [str(p) for p in Path(dir_path).glob(jpg_path)]
        self._image_paths.sort()

        self._imgs = [self.get_img(str(p)) for p in self._image_paths]
        self._len = len(self._imgs)

    def __len__(self):
        return self._len

    def get_img(self, path):
        img = Image.open(path)
        img = img.convert("RGB")
        # image = image.transform(
        #     image.size, Image.AFFINE, (1, 0, 15, 0, 1, 15), Image.BILINEAR
        # )
        # image2 = np.array(image)

        img = img.resize((128 + self.dsize, 96 + self.dsize))

        img = random_crop_image(img, self.size, test=self.test)
        img = torchvision.transforms.ToTensor()(img)
        return img

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
    def __init__(self, dir_path, img_size=(128, 96), is_test=False, dsize=5, noise=0):
        super(MyDataSetForAttention, self).__init__()

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
            img_paths.sort()
            self._class_id += [class_id for _ in range(len(img_paths))]
            class_id += 1
            self._image_paths += img_paths
        self._imgs = [self.get_img(str(p)) for p in self._image_paths]
        self._len = len(self._imgs)

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
            [img, self._class_id[index]],
        )

    @classmethod
    def save_img(self, tensor, filename):
        # hsv = torchvision.transforms.functional.to_pil_image(tensor, "HSV")
        # rgb = hsv.convert("RGB")
        # rgb.save(filename)
        torchvision.utils.save_image(tensor, filename)


"""
    def transform(self, img):
        img = random_crop_image(img, self.size, test=self.test)
        img = np.asarray(img)
        img_distort = np.copy(img)
        if self.distort:
            # if random.randrange(2):  # add random at twice
            # img_distort = random_distort(img_distort)

        img_distort = img_distort.astype(np.float32).transpose(2, 0, 1)
        img = img.astype(np.float32).transpose(2, 0, 1)
        img_distort /= 255.0
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
"""

if __name__ == "__main__":
    import os

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../data/"
    DATA_PATH = DATA_DIR + "validate"
    CORRECT_DIR = DATA_DIR + "datacheck/"

    dataset = MyDataSetForAttention(DATA_PATH, is_test=False, dsize=5, noise=0.01)
    testloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        num_workers=4,
    )
    for i, (inputs, labels) in enumerate(testloader):
        # inputs, labels = inputs.cpu(), labels.cpu()

        for j, img in enumerate(inputs):
            dataset.save_img(img, CORRECT_DIR + "{}_{}_input.jpg".format(i, j))
            # dataset.save_img(labels[0][j], CORRECT_DIR + "{}_{}_label.jpg".format(i, j))
    print(i)
