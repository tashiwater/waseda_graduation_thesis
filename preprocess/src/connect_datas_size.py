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
        RESULT_DIR = "/home/assimilation/TAKUMI_SHIMIZU/waseda_graduation_thesis/NN/data/1123_MTRNN_size/"

    motion_datas = read_csvs(INPUT_DIR + "motion_csv/")
    tactile_datas = read_csvs(INPUT_DIR + "tactile_raw/")
    image_feature_datas = tactile_datas
    # image_feature_datas = read_csvs(INPUT_DIR + "image_feature/")
    EACH_SAMPLE = 4
    start_index = 0
    first_step = 0
    sequence_num = 160  # len(image_feature_datas[0])
    test_span = 4

    motion_before_scale = [
        [1.7873218429046445, 2.058947496177859],
        [-0.6268000628434227, 0.005986479242794388],
        [-0.35454616986905824, 0.02745402833492248],
        [0.8632747860467361, 1.5960337728650325],
        [0.21390754553441596, 0.7578219292580048],
        [-0.16929693411449862, 0.2787116258541295],
        [-1.3753195038347534, -1.0480527588016495],
        [21.323997497558594, 43.88399887084961],
        [-12.402000427246096, 15.775500297546387],
        [-5.519999504089356, 13.943999290466307],
        [-6.23199987411499, 9.94333267211914],
        [-1.93066668510437, 2.7149999141693115],
        [-7.15000057220459, 1.900000095367432],
        [-1.807666778564453, 5.485333442687988],
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
    tactile_before_scale = [
        [0.0, 0.7419354319572449],
        [0.0, 0.6354838609695435],
        [0.0, 0.5774193406105042],
        [0.0, 0.745161235332489],
        [0.0, 0.699999988079071],
        [0.0032258064020425077, 0.6935483813285828],
        [0.006451612804085015, 0.625806450843811],
        [0.0, 0.7032257914543152],
        [0.0, 0.7290322184562683],
        [0.0, 0.6806451678276062],
        [0.0032258064020425077, 0.6064516305923462],
        [0.0, 0.7548387050628662],
        [0.0, 0.8096774220466614],
        [0.0, 0.8258064389228821],
        [0.0, 0.6645161509513855],
        [0.0, 0.5870967507362366],
    ]  # [[0, 1] for _ in range(tactile_datas[0].shape[1])]
    # image_before_scale = [[-0.25, 0.25] for _ in range(image_feature_datas[0].shape[1])]
    # image_before_scale = [[0,1] for _ in range(image_feature_data.shape[1])]
    size_list = [
        [160, 115, 0],
        [110, 75, 0],
        [70, 70, 0],
        [130, 130, 1],
        [140, 140, 1],
        [80, 80, 1],
    ]
    size_list = np.array(size_list)
    size_before_scale = [[70, 160], [70, 140], [0, 1]]
    for i, (motion_data, tactile_data, img) in enumerate(
        zip(motion_datas, tactile_datas, image_feature_datas)
    ):
        motion_preprocessed = get_meaned_data(
            motion_data, first_step, sequence_num, EACH_SAMPLE, start_index
        )
        motion_preprocessed = sigmoid_normalize(
            motion_preprocessed, motion_before_scale
        )
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
        target_size = size_list[i // test_span].reshape(1, -1)
        size_preprocessed = get_meaned_data(
            target_size, first_step, sequence_num, 1, start_index
        )
        size_preprocessed = sigmoid_normalize(size_preprocessed, size_before_scale)

        cs_num = 30
        for k in range(cs_num):
            size_preprocessed[k] = [0, 0, 0]

        connected_data = np.block(
            # [motion_preprocessed, tactile_preprocessed, img_preprocessed]
            [motion_preprocessed, tactile_preprocessed, size_preprocessed]
        )
        header = (
            ["position{}".format(i) for i in range(motion_preprocessed.shape[1] // 2)]
            + ["torque{}".format(i) for i in range(motion_preprocessed.shape[1] // 2)]
            + ["tactile{}".format(i) for i in range(tactile_preprocessed.shape[1])]
            + ["size{}".format(i) for i in range(size_preprocessed.shape[1])]
            # + ["image{}".format(i) for i in range(img.shape[1])]
        )
        df = pd.DataFrame(data=connected_data, columns=header)
        # test_span = 4
        # if i % test_span == 0:
        #     file_path = RESULT_DIR + "test/{:03}.csv".format(i)
        # else:
        #     file_path = RESULT_DIR + "train/{:03}.csv".format(i)
        if dump_directly:

            if i % test_span == 0:
                file_path = RESULT_DIR + "test/{:03}.csv".format(i)
            else:
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
