#!/usr/bin/env python
# coding:utf-8

import random
import cv2
import numpy as np
from PIL import ImageEnhance


def color_argumentation(img, pil_range=[0.8, 1.2]):
    img = ImageEnhance.Brightness(img)
    img = img.enhance(random.uniform(pil_range[0], pil_range[1]))
    img = ImageEnhance.Contrast(img)
    img = img.enhance(random.uniform(pil_range[0], pil_range[1]))
    img = ImageEnhance.Color(img)
    img = img.enhance(random.uniform(pil_range[0], pil_range[1]))
    return img
