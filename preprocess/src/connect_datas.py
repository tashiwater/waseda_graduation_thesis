#!/usr/bin/env python3
# coding: utf-8
from pathlib import Path
import numpy as np
import os
import pandas as pd


def read_csvs(folder_name):
    paths = [str(p) for p in Path(folder_name).glob("./*/*.csv")]
    paths.sort()
    ret = [np.loadtxt(path, delimiter=",") for path in paths]
    return ret


def get_meaned_data(data, sequence_num, each_sample):
    can_change_num = min(sequence_num, int(len(data) / each_sample))
    ret = [
        data[i * each_sample : i * each_sample + each_sample].mean(axis=0)
        for i in range(can_change_num)
    ]
    for i in range(sequence_num - len(ret)):
        ret.append(ret[-1])
    ret = np.array(ret)
    return ret


def min_max_normalization(data, indataRange, outdataRange):
    data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
    data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    return data


def sigmoid_normalize(data, before_scale):
    for i, scale in enumerate(before_scale):
        data[:, i] = min_max_normalization(data[:, i], scale, [-0.9, 0.9])
    return data


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../data/"
    INPUT_DIR = DATA_DIR + "inputs/"
    RESULT_DIR = DATA_DIR + "connected/"
    motion_datas = read_csvs(INPUT_DIR + "motion_csv/")
    tactile_datas = read_csvs(INPUT_DIR + "tactile/")
    image_feature_datas = read_csvs(INPUT_DIR + "image_feature/")
    EACH_SAMPLE = 4

    sequence_num = len(image_feature_datas[0])
    motion_before_scale = [
        [-1.309, 4.451],
        [-2.094, 0.140],
        [-2.880, 2.880],
        [-0.524, 2.269],
        [-2.880, 2.880],
        [-1.396, 1.396],
        [-1.396, 0.087],
        [-2, 40],
        [-40, 15],
        [-5, 15],
        [-10, 15],
        [-5, 5],
        [-5, 5],
        [-5, 5],
    ]
    tactile_before_scale = [[0, 1] for _ in range(tactile_datas[0].shape[1])]
    image_before_scale = [[0, 1] for _ in range(image_feature_datas[0].shape[1])]
    # image_before_scale = [[0,1] for _ in range(image_feature_data.shape[1])]
    for i, (motion_data, tactile_data, image_feature_data) in enumerate(
        zip(motion_datas, tactile_datas, image_feature_datas)
    ):
        motion_preprocessed = get_meaned_data(motion_data, sequence_num, EACH_SAMPLE)
        motion_preprocessed = sigmoid_normalize(
            motion_preprocessed, motion_before_scale
        )
        tactile_preprocessed = get_meaned_data(tactile_data, sequence_num, EACH_SAMPLE)
        tactile_preprocessed = sigmoid_normalize(
            tactile_preprocessed, tactile_before_scale
        )
        image_feature_datas_preprocessed = sigmoid_normalize(
            image_feature_data, image_before_scale
        )
        connected_data = np.block(
            [motion_preprocessed, tactile_preprocessed, image_feature_data]
        )
        header = (
            ["position{}".format(i) for i in range(motion_preprocessed.shape[1] // 2)]
            + ["torque{}".format(i) for i in range(motion_preprocessed.shape[1] // 2)]
            + ["tactile{}".format(i) for i in range(tactile_preprocessed.shape[1])]
            + ["image{}".format(i) for i in range(image_feature_data.shape[1])]
        )
        df = pd.DataFrame(data=connected_data, columns=header)
        df.to_csv(RESULT_DIR + "{:02}.csv".format(i + 1), index=False)
        #     connected_datas.append(connected_data)
        # connected_datas = np.ndarray(connected_datas)
        # # [TODO] add title
        # np.savetxt(
        #     RESULT_DIR + "{:02}.csv".format(i + 1), connected_data, delimiter=",",
        # )
