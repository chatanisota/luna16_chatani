"""
    前処理されたデータセットから、
    可視化に使用するサンプル画像だけ取得するライブラリ
"""

from settings import *
import os
import tqdm
import numpy as np

def get_sample_malignants():

    m_list = [slice_png_malignant_dir + s + '.png' for s in visualize_sample_malignant_numbers]
    m_list = [slice_npy_malignant_dir + os.path.splitext(os.path.basename(p))[0] + '.npy' for p in m_list]
    data = []
    for x in tqdm.tqdm(m_list):
        m_array = np.load(x)
        data.append(m_array)
    data = np.array(data)

    return data, m_list

def get_sample_benigns():

    b_list = [slice_png_benign_dir + s + '.png' for s in visualize_sample_benign_numbers]
    b_list = [slice_npy_benign_dir + os.path.splitext(os.path.basename(p))[0] + '.npy' for p in b_list]
    data = []

    for x in tqdm.tqdm(b_list):
        b_array = np.load(x)
        data.append(b_array)
    data = np.array(data)

    return data, b_list
