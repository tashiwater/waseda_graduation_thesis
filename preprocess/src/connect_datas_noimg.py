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
        RESULT_DIR = CURRENT_DIR + "/../../NN/data/MTRNN/1226/normal/"

    motion_datas = read_csvs(INPUT_DIR + "motion_csv/")
    tactile_datas = read_csvs(INPUT_DIR + "tactile_raw/")
    image_feature_datas = tactile_datas
    # image_feature_datas = read_csvs(INPUT_DIR + "image_feature/")
    EACH_SAMPLE = 4
    start_index = 0
    first_step = 0
    sequence_num = 115  # len(image_feature_datas[0])

    motion_before_scale = [
        [-0.04270818229521933, 0.18957765482950872],
        [-0.29682817864355704, 0.2959554940438517],
        [-0.178651903299404, 0.07360053468975894],
        [-0.8050505611875245, 0.06068505228547205],
        [-8.725181521368475e-05, 0.3774274174542602],
        [-0.23530529587914603, 0.20772908254564854],
        [-0.0026004436464077685, 0.3106860372443254],
        [-7.007999420166016, 17.700000762939453],
        [-9.652500629425049, 11.544000625610352],
        [-4.583999872207642, 5.915999412536621],
        [-4.623332977294922, 12.780665636062622],
        [-0.9049999713897705, 2.8959999084472656],
        [-1.9500001072883606, 2.0000000596046448],
        [-1.4336667656898499, 3.1166669726371765],
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
        [0.0, 0.7096773982048035],
        [0.0, 0.6193548440933228],
        [0.0, 0.5419355034828186],
        [0.0, 0.7193548083305359],
        [-0.03548387438058853, 0.6451613157987595],
        [-0.04838709160685539, 0.6645161211490631],
        [-0.04838710278272629, 0.5645161587744951],
        [-0.05806452035903931, 0.6322580575942993],
        [-0.49999999068677425, 0.34838709235191345],
        [-0.641935515217483, 0.2709677517414093],
        [-0.4709677454084158, 0.24838706851005554],
        [-0.5129032013937831, 0.3838709592819214],
        [-0.10000000149011612, 0.7677419185638428],
        [-0.08387096971273422, 0.7516129016876221],
        [-0.06129032373428345, 0.5580645203590393],
        [-0.029032258316874504, 0.5709677278064191],
    ]
    # tactile_before_scale = [[0, 1] for _ in range(tactile_datas[0].shape[1])]
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
