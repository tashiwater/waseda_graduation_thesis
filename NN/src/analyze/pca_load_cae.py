#!/usr/bin/env python3
# coding: utf-8
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from pca_load_base import PcaLoadBase

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = current_dir + "/../../data/MTRNN/0106/all/"
    pca_load = PcaLoadBase()
    pca_load.set_params(4, 1, 30, 15)
    pca_load.load(data_dir + "/train/", 3, True)
    pca_load.load(data_dir + "/test/", 1, False)
    pca_load.run()
