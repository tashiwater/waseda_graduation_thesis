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
        RESULT_DIR = "/home/assimilation/TAKUMI_SHIMIZU/waseda_graduation_thesis/NN/data/MTRNN_cs/"
    motion_datas = read_csvs(INPUT_DIR + "motion_csv/")
    tactile_datas = read_csvs(INPUT_DIR + "tactile_raw/")
    image_feature_datas = read_csvs(INPUT_DIR + "image_feature/")
    EACH_SAMPLE = 4
    start_index = 0
    first_step = 0
    sequence_num = 210  # len(image_feature_datas[0])

    motion_before_scale = [
        [1.6117068753543122, 2.1200514637028007],
        [-0.6368531630633356, 0.11388273535710568],
        [-0.5634097049436209, 0.19519762380936195],
        [0.7681368847920997, 1.688274487748799],
        [0.16900022467955014, 0.9541540264022268],
        [-0.8862432854471518, 0.29183651594388066],
        [-1.2531987873105703, -0.5639856801083778],
        [20.052000045776367, 58.91999435424805],
        [-29.796001434326172, 13.532999992370605],
        [-5.3279995918273935, 25.24799728393555],
        [-9.5, 18.227333068847656],
        [-3.86133337020874, 2.5339999198913574],
        [-9.450000762939453, 4.800000190734863],
        [-2.119333267211914, 8.414999961853027],
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
    # tactile_before_scale = [[0, 1] for _ in range(tactile_datas[0].shape[1])]
    tactile_before_scale = [
        [0.0, 0.7548387050628662],
        [0.0, 0.72258061170578],
        [0.0, 0.6354838609695435],
        [0.0, 0.7935483455657959],
        [0.0, 0.6645161509513855],
        [0.0, 0.7064515948295593],
        [0.0, 0.6419354677200317],
        [0.0, 0.6870967745780945],
        [0.0, 0.7161290049552917],
        [0.0, 0.6580645442008972],
        [0.0, 0.5193548202514648],
        [0.0, 0.7161290049552917],
        [0.0, 0.79677414894104],
        [0.0, 0.8129032254219055],
        [0.0, 0.6451613306999207],
        [0.0, 0.5903225541114807],
    ]
    # image_before_scale = [[-0.25, 0.25] for _ in range(image_feature_datas[0].shape[1])]
    image_before_scale = [
        [0.0, 0.4972033500671387],
        [0.0, 0.4519493579864502],
        [0.0, 0.4811547100543976],
        [0.0, 0.40289184451103205],
        [0.0, 0.4973227083683014],
        [0.008948994800448418, 0.44961935281753534],
        [0.0035407552495598797, 0.4977116882801056],
        [0.0, 0.5266909599304199],
        [0.0, 0.4620552659034728],
        [0.0, 0.4891946613788604],
        [0.0, 0.3642278611660004],
        [0.0, 0.4532333612442017],
        [0.0, 0.5029321908950806],
        [0.05297283828258514, 0.3900356590747833],
        [0.0, 0.4649696946144104],
    ]
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
        if dump_directly:
            test_span = 4
            if i % test_span == 0:
                file_path = RESULT_DIR + "test/{:03}.csv".format(i)
            else:
                file_path = RESULT_DIR + "train/{:03}.csv".format(i)
        else:
            file_path = RESULT_DIR + "{:03}.csv".format(i)

        df.to_csv(file_path, index=False)
        print(file_path)
        #     connected_datas.append(connected_data)
        # connected_datas = np.ndarray(connected_datas)
