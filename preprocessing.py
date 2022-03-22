"""
    前処理用のプログラミング

    LUNA16のサブセット0-9と、秘伝のCSV(※)から
    肺結節をクロップし、良悪性に分類する
    論文の「前処理」参照

    元々のソースファイルは2019年にChenさんからもらったもの。
    おそらくhttps://github.com/GorkemP/LUNA16_Challangeが原型であると考えられる。

    ※BMnodule.csv
    LUNA16にはBMnodule.csvはなく、Chenさんからもらったもの。
    どこから持ってきたのかは正確には不明
    元々のファイル名は、「annotationdetclssgm_doctor.csv」であり、

    Zhu, Wentao, Chaochun Liu, Wei Fan, and Xiaohui Xie. "DeepLung: Deep 3D Dual Path Nets for Automated Pulmonary Nodule Detection and Classification." IEEE WACV, 2018.
    https://github.com/wentaozhu/DeepLung/tree/master/nodcls

    から、取得してきた可能性が高い。(実際に、Chenさんはこの論文を読んでいたと思う。)
    LUNA16は、LIDC-IDRIのデータを選別したものであり、LIDC-IDRIは良悪性情報を持つことから、
    Wentao氏は、LIDC-IDRIとLUNA16を対応付けしてLUNA16のclass分類を行ったものだと考えられる。
"""

import cv2
import pandas as pd
import os
import tqdm
import numpy as np
from glob import glob
import shutil
import SimpleITK as sitk
from scipy import ndimage
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet)
from settings import *

IMAGE_SIZE = 48
RESIZE_SPACING = [0.69, 0.69] # 全spacingの平均値、0.69mm x 0.69mmにリサンプリング
P = int(IMAGE_SIZE / 2) # padding

# HUを-1000 HUから 400 HUまでに制限
def truncate_hu(image_array):
    image_array[image_array > 400] = 400
    image_array[image_array <-1000] = -1000
    return image_array

# 標準化
def normalaization(image_array):
    max = 400
    min = -1000
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    #mean = image_array.mean()
    #image_array = image_array - mean
    return image_array

# 標本化
def resampling(scan_slice, spacing):
    resize_factor = spacing / RESIZE_SPACING
    new_real_shape = scan_slice.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / scan_slice.shape
    scan_slice = ndimage.zoom(scan_slice, zoom=(real_resize[0], real_resize[1]), order=1)
    return scan_slice

def renew_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)

# ファイルを作り直す
#renew_folder(npy_benign_dir)
#renew_folder(npy_malignant_dir)
renew_folder(slice_png_benign_dir)
renew_folder(slice_png_malignant_dir)
renew_folder(slice_npy_benign_dir)
renew_folder(slice_npy_malignant_dir)

df = pd.read_csv(annotation_file, sep=',')

# Read subsets path
subset_path_list = []
for i in tqdm.tqdm(range(10)):
    print(base_luna16_dir +'subset'+str(i)+"/")
    subset_path_list.append(base_luna16_dir +'subset'+str(i)+"/")

for subset_id, subset_path in enumerate(subset_path_list):
    mhd_path_list = glob(subset_path + "*.mhd")

    for scan_id, mhd_path in enumerate(tqdm.tqdm(mhd_path_list)):
        mhd_path_filename = mhd_path.split('\\')[1]
        seriesuid = mhd_path_filename[:-4] #.mdh = :-4
        df_nodules = df[df['seriesuid']==seriesuid]

        # if you find nodule in mhd
        if df_nodules.shape[0] <= 0:
            continue

        scan_itk    = sitk.ReadImage(mhd_path)
        scan_spacing = np.array(scan_itk.GetSpacing())    # spacing of voxels in world coor. (mm)
        scan_origin = np.array(scan_itk.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        scan_array  = sitk.GetArrayFromImage(scan_itk).transpose(2,1,0)  # indexes are z,y,x (notice the ordering)

        for nodule_id, df_nodule in df_nodules.iterrows():
            nodule_pos = [df_nodule['coordX'], df_nodule['coordY'], df_nodule['coordZ']]
            c = np.rint(( nodule_pos - scan_origin) / (RESIZE_SPACING[0], RESIZE_SPACING[1], scan_spacing[2] ))  # nodule center in voxel space (still x,y,z ordering)
            c = [int(c[0]), int(c[1]), int(c[2])]
            scan_slice = scan_array[:,:,c[2]]

            # 標本化と正規化
            scan_slice = resampling(scan_slice, scan_spacing[0:1])
            scan_slice  = normalaization(truncate_hu(scan_slice))

            # クロップがCT外を参照している時は無視する
            if(c[0]-P<0 or c[0]+P>=scan_slice.shape[0]):
                continue
            if(c[1]-P<0 or c[1]+P>=scan_slice.shape[1]):
                continue
            crop_array = scan_slice[ c[0]-P:c[0]+P, c[1]-P:c[1]+P ]

            file_code = str(subset_id)+"_"+str(scan_id)+"_"+str(nodule_id)

            if(df_nodule['class'] == 1):
                np.save(slice_npy_benign_dir+file_code, crop_array)
                cv2.imwrite(slice_png_benign_dir+file_code+".png", crop_array*255)
            else:
                np.save(slice_npy_malignant_dir+file_code, crop_array)
                cv2.imwrite(slice_png_malignant_dir+file_code+".png", crop_array*255)
