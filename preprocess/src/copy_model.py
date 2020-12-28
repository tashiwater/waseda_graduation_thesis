#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
from pathlib import Path

model_dir = "/home/assimilation/TAKUMI_SHIMIZU/model/MTRNN/1226/normal/"
cf_list = [50, 60, 70, 80, 90, 100, 110]
cs_list = [6, 8, 10, 12]

for cf in cf_list:
    for cs in cs_list:
        model_path = model_dir + "{}_{}/".format(cf, cs)
        num = 5000
        output = model_dir + "{}/{}_{}.pth".format(num, cf, cs)
        paths = [str(p) for p in Path(model_path).glob("./*{}.pth".format(num))]
        if len(paths) != 1:
            print("there are {} finish.pth".format(len(paths)))
            continue
            raise FileExistsError("there are {} finish.pth".format(len(paths)))
        shutil.copy(paths[0], output)
        # output = model_dir + "2000/{}_{}.csv".format(cf, cs)
        # paths = [str(p) for p in Path(model_path).glob("./*finish.csv")]
        # if len(paths) != 1:
        #     raise FileExistsError("there are {} finish.pth".format(len(paths)))
        # shutil.copy(paths[0], output)
        print(paths[0])
