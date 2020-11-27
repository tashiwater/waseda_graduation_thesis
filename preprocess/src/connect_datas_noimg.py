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
        RESULT_DIR = "/home/assimilation/TAKUMI_SHIMIZU/waseda_graduation_thesis/NN/data/MTRNN_cs/"

    motion_datas = read_csvs(INPUT_DIR + "motion_csv/")
    tactile_datas = read_csvs(INPUT_DIR + "tactile_raw/")
    image_feature_datas = tactile_datas
    # image_feature_datas = read_csvs(INPUT_DIR + "image_feature/")
    EACH_SAMPLE = 4
    start_index = 0
    first_step = 0
    sequence_num = 160  # len(image_feature_datas[0])

    motion_before_scale = [
        [1.5525751117746291, 2.159618048018108],
        [-0.6268000628434227, 0.12201597106955885],
        [-0.5634621026380682, 0.1518261978894153],
        [0.6376909269632473, 1.6881173612444842],
        [0.17943730279438197, 0.9541191389919468],
        [-0.8867668629174613, 0.2787116258541295],
        [-1.3753195038347534, -0.4766494093985876],
        [17.15999984741211, 56.85599899291992],
        [-16.321500778198242, 15.775500297546387],
        [-5.519999504089356, 14.615999221801758],
        [-12.09666633605957, 15.99799919128418],
        [-1.93066668510437, 2.7149999141693115],
        [-7.15000057220459, 2.700000047683716],
        [-1.807666778564453, 6.295667171478272],
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
        [0.0, 0.6580645442008972],
        [0.0, 0.5774193406105042],
        [0.0, 0.745161235332489],
        [0.0, 0.699999988079071],
        [0.0, 0.6935483813285828],
        [0.0, 0.625806450843811],
        [0.0, 0.7032257914543152],
        [0.0, 0.7290322184562683],
        [0.0, 0.6806451678276062],
        [0.0032258064020425077, 0.6064516305923462],
        [0.0, 0.7548387050628662],
        [0.0, 0.8193548321723938],
        [0.0, 0.8258064389228821],
        [0.0, 0.6645161509513855],
        [0.0, 0.5870967507362366],
    ]  # [[0, 1] for _ in range(tactile_datas[0].shape[1])]
    # image_before_scale = [[-0.25, 0.25] for _ in range(image_feature_datas[0].shape[1])]
    # image_before_scale = [[0,1] for _ in range(image_feature_data.shape[1])]
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
            test_span = 4
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
