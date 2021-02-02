#!/usr/bin/env python3
# coding: utf-8
from pathlib import Path
import numpy as np
import os
import pandas as pd


def read_csvs(folder_name):
    paths = [str(p) for p in Path(folder_name).glob("./*.csv")]
    paths.sort()
    ret = [np.loadtxt(path, delimiter=",") for path in paths]
    return ret


def get_meaned_data(data, first_step, sequence_num, each_sample, start_index):
    can_change_num = min(sequence_num, int(len(data[start_index:]) / each_sample))
    ret = [data[0] for _ in range(first_step)]

    for i in range(can_change_num):
        meaned = data[
            start_index + i * each_sample : start_index + i * each_sample + each_sample
        ].mean(axis=0)
        ret.append(meaned)
    # if each_sample == 1:
    #     ret = list(data[:can_change_num])
    # else:
    #     ret = [
    #         data[i * each_sample : i * each_sample + each_sample].mean(axis=0)
    #         for i in range(can_change_num)
    #     ]
    for i in range(first_step + sequence_num - len(ret)):
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
    dump_directly = False
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../data/"
    INPUT_DIR = DATA_DIR + "connect_input/"
    RESULT_DIR = DATA_DIR + "connected/"
    if dump_directly:
        RESULT_DIR = CURRENT_DIR + "/../../NN/data/MTRNN/0106/all/"

    motion_datas = read_csvs(INPUT_DIR + "motion_csv/")
    tactile_datas = read_csvs(INPUT_DIR + "tactile_raw/")
    image_feature_datas = read_csvs(INPUT_DIR + "image_feature/")
    EACH_SAMPLE = 4
    start_index = 0
    first_step = 0
    sequence_num = 140  # len(image_feature_datas[0])

    motion_before_scale = [
        [-0.18788481648259991, 0.042219492235189726],
        [-0.2643824543194445, 0.35691982870388905],
        [-0.5818403877102672, 0.03354522767509434],
        [-0.16716768688870987, 0.4605400478332392],
        [-0.028553580969988945, 0.7077135546627552],
        [-0.4690223324184392, 0.00034907383988214136],
        [-0.08326965701475286, 0.2065073698791423],
        [-9.911998748779297, 12.623998641967773],
        [-13.767000198364258, 6.922499656677246],
        [-2.783999979496002, 9.923999547958374],
        [-10.93133282661438, 6.764000415802002],
        [-1.6893332302570343, 1.7496666014194489],
        [-3.9500001668930054, 2.849999964237213],
        [-1.9946666955947876, 2.4310001134872437],
    ]

    # motion_before_scale = [
    #     [-1.309, 4.451],
    #     [-2.094, 0.140],
    #     [-2.880, 2.880],
    #     [-0.524, 2.269],
    #     [-2.880, 2.880],
    #     [-1.396, 1.396],
    #     [-1.396, 0.087],
    #     [-2, 40],
    #     [-40, 15],
    #     [-5, 15],
    #     [-10, 15],
    #     [-5, 5],
    #     [-5, 5],
    #     [-5, 5],
    # ]
    # tactile_before_scale = [
    #     [0.0, 0.7161290049552917],
    #     [0.0, 0.6193548440933228],
    #     [0.0, 0.5129032135009766],
    #     [0.0, 0.7645161151885986],
    #     [-0.029032262042164803, 0.6451613157987595],
    #     [-0.03870967775583267, 0.6645161211490631],
    #     [-0.029032256454229355, 0.5645161587744951],
    #     [-0.03870967775583267, 0.6258064769208431],
    #     [-0.43548389966599643, 0.43225806951522827],
    #     [-0.6161290286108851, 0.24193552136421204],
    #     [-0.4258064776659012, 0.23225805163383484],
    #     [-0.48709677439182997, 0.4322580099105835],
    #     [-0.10967742651700974, 0.7774193286895752],
    #     [-0.08709677122533321, 0.7516129016876221],
    #     [-0.06129032373428345, 0.5580645203590393],
    #     [-0.032258064951747656, 0.5193548081442714],
    # ]
    tactile_before_scale = [[0, 1] for _ in range(tactile_datas[0].shape[1])]
    # image_before_scale = [[-0.25, 0.25] for _ in range(image_feature_datas[0].shape[1])]
    image_before_scale = [[0, 1] for _ in range(image_feature_datas[0].shape[1])]
    for i, (motion_data, tactile_data, img) in enumerate(
        zip(motion_datas, tactile_datas, image_feature_datas)
    ):
        motion_data -= motion_data[0]
        motion_preprocessed = get_meaned_data(
            motion_data, first_step, sequence_num, EACH_SAMPLE, start_index
        )
        motion_preprocessed = sigmoid_normalize(
            motion_preprocessed, motion_before_scale
        )
        tactile_data -= tactile_data[0]  # calibration
        tactile_preprocessed = get_meaned_data(
            tactile_data, first_step, sequence_num, EACH_SAMPLE, start_index
        )
        tactile_preprocessed = sigmoid_normalize(
            tactile_preprocessed, tactile_before_scale
        )
        img_preprocessed = get_meaned_data(
            img, first_step, sequence_num, 1, start_index
        )
        img_preprocessed = sigmoid_normalize(img_preprocessed, image_before_scale)
        connected_data = np.block(
            [motion_preprocessed, tactile_preprocessed, img_preprocessed]
            # [motion_preprocessed, tactile_preprocessed]
        )
        header = (
            ["position{}".format(i) for i in range(motion_preprocessed.shape[1] // 2)]
            + ["torque{}".format(i) for i in range(motion_preprocessed.shape[1] // 2)]
            + ["tactile{}".format(i) for i in range(tactile_preprocessed.shape[1])]
            + ["image{}".format(i) for i in range(img.shape[1])]
        )
        df = pd.DataFrame(data=connected_data, columns=header)
        # test_span = 4
        # if i % test_span == 0:
        #     file_path = RESULT_DIR + "test/{:03}.csv".format(i)
        # else:
        #     file_path = RESULT_DIR + "train/{:03}.csv".format(i)
        if dump_directly:
            test_span = 4
            if i % test_span == 0:
                file_path = RESULT_DIR + "test/{:03}.csv".format(i)
            else:
                file_path = RESULT_DIR + "train/{:03}.csv".format(i)
            # file_path = RESULT_DIR + "train/{:03}.csv".format(i)
        else:
            file_path = RESULT_DIR + "{:03}.csv".format(i)
        # file_path = RESULT_DIR + "{:03}.csv".format(i)
        df.to_csv(file_path, index=False)
        print(file_path)
        #     connected_datas.append(connected_data)
        # connected_datas = np.ndarray(connected_datas)
        # # [TODO] add title
        # np.savetxt(
        #     RESULT_DIR + "{:02}.csv".format(i + 1), connected_data, delimiter=",",
        # )
