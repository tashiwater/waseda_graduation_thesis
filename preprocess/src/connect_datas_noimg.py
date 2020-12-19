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
    dump_directly = True
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../data/"
    INPUT_DIR = DATA_DIR + "connect_input/"
    RESULT_DIR = DATA_DIR + "connected/"
    if dump_directly:
        RESULT_DIR = CURRENT_DIR + "/../../NN/data/MTRNN/1218/noimg/"

    motion_datas = read_csvs(INPUT_DIR + "motion_csv/")
    tactile_datas = read_csvs(INPUT_DIR + "tactile_raw/")
    image_feature_datas = tactile_datas
    # image_feature_datas = read_csvs(INPUT_DIR + "image_feature/")
    EACH_SAMPLE = 4
    start_index = 0
    first_step = 0
    sequence_num = 140  # len(image_feature_datas[0])

    motion_before_scale = [
        [-0.2547831567492811, 0.3522597794941982],
        [-0.19786794214487075, 0.5512273242084056],
        [-0.47539278851585526, 0.28305750527897533],
        [-0.6276378933223615, 0.38781218109842874],
        [-0.22455405630184733, 0.6033428733829775],
        [-0.9111142056498871, 0.2545039097448242],
        [-0.17498679442820975, 0.6055594556477641],
        [-14.483999252319336, 24.43199920654297],
        [-20.572500705718994, 11.095499992370605],
        [-6.299999475479126, 13.763999283313751],
        [-13.971332907676697, 14.211999297142029],
        [-1.749666690826416, 2.956333249807358],
        [-6.40000057220459, 3.5500000715255737],
        [-2.742666721343994, 5.298333764076233],
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
    tactile_before_scale = [[0, 1] for _ in range(tactile_datas[0].shape[1])]
    # image_before_scale = [[-0.25, 0.25] for _ in range(image_feature_datas[0].shape[1])]
    # image_before_scale = [[0,1] for _ in range(image_feature_data.shape[1])]
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
        # img_preprocessed = get_meaned_data(
        #     img, first_step, sequence_num, 1, start_index
        # )
        # img_preprocessed = sigmoid_normalize(img_preprocessed, image_before_scale)
        connected_data = np.block(
            # [motion_preprocessed, tactile_preprocessed, img_preprocessed]
            [motion_preprocessed, tactile_preprocessed]
        )
        header = (
            ["position{}".format(i) for i in range(motion_preprocessed.shape[1] // 2)]
            + ["torque{}".format(i) for i in range(motion_preprocessed.shape[1] // 2)]
            + ["tactile{}".format(i) for i in range(tactile_preprocessed.shape[1])]
            # + ["image{}".format(i) for i in range(img.shape[1])]
        )
        df = pd.DataFrame(data=connected_data, columns=header)
        # test_span = 4
        # if i % test_span == 0:
        #     file_path = RESULT_DIR + "test/{:03}.csv".format(i)
        # else:
        #     file_path = RESULT_DIR + "train/{:03}.csv".format(i)
        if dump_directly:
            # test_span = 4
            # if i % test_span == 0:
            #     file_path = RESULT_DIR + "test/{:03}.csv".format(i)
            # else:
            file_path = RESULT_DIR + "train/{:03}.csv".format(i)
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
