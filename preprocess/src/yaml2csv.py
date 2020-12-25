#!/usr/bin/env python3
# coding: utf-8
import yaml
import os
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"

for class_num in range(1):
    folder_name = ""  # "motion{}/".format(class_num + 1)
    YAML_DIR = (
        "/home/assimilation/TAKUMI_SHIMIZU/wiping_ws/src/wiping/data/1225/motion_yaml/"
        + folder_name
    )
    RESULT_DIR = DATA_DIR + "connect_input/motion_csv/" + folder_name
    paths = [str(p) for p in Path(YAML_DIR).glob("./*.yaml")]
    paths.sort()
    # print(paths)
    for i, path in enumerate(paths):
        print(path)
        f_yaml = open(path, "r")
        f_csv = open(RESULT_DIR + "{:03}.csv".format(i), "w")
        ydata = yaml.safe_load(f_yaml)
        # {'arm_controller': {'teaching_trajectories': {'names': ['traj0'], 'traj0': [{'accelerations': [...], 'effort': [...], 'positions': [...], 'time_from_start': 0.0, 'velocities': [...]}, {'accelerations': ...]}]}}}
        # print(ydata["arm_controller"]["teaching_trajectories"]["traj0"])
        # strs = []
        for i in ydata["arm_controller"]["teaching_trajectories"]["traj0"]:
            for j in i["positions"]:
                # strs += str(j) + ","
                f_csv.write(str(j))
                f_csv.write(",")
            for j in i["effort"][:-1]:
                # strs += str(j) + ","
                f_csv.write(str(j))
                f_csv.write(",")
            f_csv.write(str(i["effort"][-1]) + "\n")
            # strs += str(i["effort"][-1]) + "\n"
            # f_csv.write(str(strs))
