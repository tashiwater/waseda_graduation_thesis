#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
from pathlib import Path

model_dir = "/home/assimilation/TAKUMI_SHIMIZU/model/MTRNN/"
cf_list = [70, 80]
cs_list = [8, 10, 12, 15]

for cf in cf_list:
    for cs in cs_list:
        model_path = model_dir + "1119_{}_{}/".format(cf, cs)
        output = model_dir + "1119_10000/{}_{}.pth".format(cf, cs)
        paths = [str(p) for p in Path(model_path).glob("./*finish.pth")]
        if len(paths) != 1:
            raise FileExistsError("there are {} finish.pth".format(len(paths)))
        shutil.copy(paths[0], output)
        print(paths[0])
