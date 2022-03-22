"""
    Kerasの機能である、ジェネレーターのオリジナル関数
"""
import tqdm
import glob
import cv2
import os
import gc
import shutil
import numpy as np
from settings import *

"""
良性は「1」悪性は「0」
"""

image_size = 48
#image_size1 = 32
#image_size2 = 40
#image_size3 = 48

CENTER = int(image_size/2)

DATA_ARGUMENTATION = True
IS_3D = False

def add_gaussian_noise(src):
    mean = 0
    sigma = 0.01

    if src.ndim == 2:
        row,col= src.shape
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
    elif src.ndim == 3:
        row,col,z= src.shape
        gauss = np.random.normal(mean,sigma,(row,col,z))
        gauss = gauss.reshape(row,col,z)
    else:
        print("Error: you miss input.")
        exit()

    noisy = src + gauss

    return noisy

def create_data():
    X = []
    y = []

    m_list = glob.glob(slice_npy_malignant_dir+'*')

    for m_name in tqdm.tqdm(m_list):
        X.append(m_name)
        y.append(0)

    b_list = glob.glob(slice_npy_benign_dir+'*')

    for b_name in tqdm.tqdm(b_list):
        X.append(b_name)
        y.append(1)

    l = list(zip(X, y))
    np.random.shuffle(l)     # シャッフル
    X, y = zip(*l)

    X = np.array(X)
    y = np.array(y)

    print("ALL: ", len(X))
    print("BENIGN: "+str(len(m_list))+" MALIGNANT: "+str(len(b_list)))

    return X, y



# 一度拡張を保存する。
def generate_train(X,Y, data_argument):

    X1 = []
    YR = []


    renew_folder(tmp_dir)
    i = 0
    if data_argument == 1:
        for x, y in tqdm.tqdm(zip(X,Y)):
            m_array = np.load(x)

            x_move = []
            x_list = []

            np.save(tmp_dir+"t"+str(i)+".npy",m_array)
            x_list.append(tmp_dir+"t"+str(i)+".npy")
            np.save(tmp_dir+"t"+str(i)+"r.npy",np.rot90(m_array,1))
            x_list.append(tmp_dir+"t"+str(i)+"r.npy")
            np.save(tmp_dir+"t"+str(i)+"rr.npy",np.rot90(m_array,2))
            x_list.append(tmp_dir+"t"+str(i)+"rr.npy")
            np.save(tmp_dir+"t"+str(i)+"rrr.npy",np.rot90(m_array,3))
            x_list.append(tmp_dir+"t"+str(i)+"rrr.npy")

            for x_arg in x_list:
                X1.append(x_arg)
                YR.append(y)
            i = i + 1

    elif data_argument == 2:
        for x, y in tqdm.tqdm(zip(X,Y)):
            m_array = np.load(x)

            x_move = []
            x_list = []

            np.save(tmp_dir+"t"+str(i)+".npy",m_array)
            x_list.append(tmp_dir+"t"+str(i)+".npy")
            np.save(tmp_dir+"t"+str(i)+"r.npy",np.rot90(m_array,1))
            x_list.append(tmp_dir+"t"+str(i)+"r.npy")
            np.save(tmp_dir+"t"+str(i)+"rr.npy",np.rot90(m_array,2))
            x_list.append(tmp_dir+"t"+str(i)+"rr.npy")
            np.save(tmp_dir+"t"+str(i)+"rrr.npy",np.rot90(m_array,3))
            x_list.append(tmp_dir+"t"+str(i)+"rrr.npy")
            np.save(tmp_dir+"t"+str(i)+"f.npy",np.fliplr(m_array))
            x_list.append(tmp_dir+"t"+str(i)+"f.npy")
            np.save(tmp_dir+"t"+str(i)+"fr.npy",np.fliplr(np.rot90(m_array,1)))
            x_list.append(tmp_dir+"t"+str(i)+"fr.npy")
            np.save(tmp_dir+"t"+str(i)+"frr.npy",np.fliplr(np.rot90(m_array,2)))
            x_list.append(tmp_dir+"t"+str(i)+"frr.npy")
            np.save(tmp_dir+"t"+str(i)+"frrr.npy",np.fliplr(np.rot90(m_array,3)))
            x_list.append(tmp_dir+"t"+str(i)+"frrr.npy")

            for x_arg in x_list:
                X1.append(x_arg)
                YR.append(y)
            i = i + 1
    else:
        for x in tqdm.tqdm(X):
            m_array = np.load(x)

            np.save(tmp_dir+"t"+str(i)+".npy",m_array)
            X1.append(tmp_dir+"t"+str(i)+".npy")
            i = i + 1
        YR = Y

    l = list(zip(X1, YR))
    np.random.shuffle(l) # シャッフル
    X1,YR = zip(*l)

    X1 = np.array(X1)
    YR = np.array(YR)

    return X1,YR

def renew_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)

def generate_test(X, y):

    X1 = []

    i = 0
    for x in tqdm.tqdm(X):

        m_array = np.load(x)

        np.save(tmp_dir+"v"+str(i)+".npy",m_array)
        X1.append(tmp_dir+"v"+str(i)+".npy")
        i = i + 1

    X1 = np.array(X1)

    return X1,y

class Generator(object):

    images = []
    labels = []

    def __init__(self):
        self.reset()

    def reset(self):
        self.images = []
        self.labels = []

    def flow_from_directory(self, X, Y, batch_size=-1):
        # LabelEncode(classをint型に変換)するためのdict
        if batch_size == -1:
            batch_size = len(X)

        while True:
            # ディレクトリから画像のパスを取り出す
            for x, y in zip(X,Y):
                # 画像を読み込みRGBへの変換、Numpyへの変換を行い、配列(self.iamges)に格納
                self.images.append(np.load(x))
                # ファイル名からラベルを取り出し、配列(self.labels)に格納
                self.labels.append(y)

                # ここまでを繰り返し行い、batch_sizeの数だけ配列(self.iamges, self.labels)に格納
                # batch_sizeの数だけ格納されたら、戻り値として返し、配列(self.iamges, self.labels)を空にする
                if len(self.images) == batch_size:
                    inputs = np.asarray(self.images, dtype=np.float32)
                    targets = np.asarray(self.labels, dtype=np.float32)
                    self.reset()
                    yield inputs, targets
