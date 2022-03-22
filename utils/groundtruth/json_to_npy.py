import json
import glob
import numpy as np
import os
import csv
import tqdm
import matplotlib.pyplot as plt
import cv2
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.visualize.samples import *

def save_heatmap(imgs, heatmaps, tag):
    for i in range(heatmaps.shape[0]):
        inner_save_heatmap(i, imgs[i], heatmaps[i], tag)


def inner_save_heatmap(no, img, heatmap, tag):
    #plt.imshow(heatmap, cmap=cmap, vmax=M, vmin=-M, interpolation=interpolation)
    colorbar = True
    cmap = "Reds"
    ## 画像の上にheatmap_imgを透過率0.4で重ねる
    #plt.imshow(heatmap, cmap=cmap, vmax=M, vmin=-M, alpha=0.7)
    plt.clf()
    fig, [ax1, ax2] = plt.subplots(figsize=(8, 4), ncols=2, sharey=True)
    im1 = ax1.imshow(img, cmap="gray", vmax=1.0, vmin=0.0, alpha=0.6, aspect="auto")
    im1 = ax1.imshow(heatmap, cmap=cmap, vmax=1, vmin=0, alpha=0.4, aspect="auto")
    im2 = ax2.imshow(heatmap, cmap=cmap, vmax=1, vmin=0, aspect="auto")

    fig.subplots_adjust(wspace=0.07)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if colorbar:
        fig.colorbar(im2, cax=cax)

    fig.suptitle(tag.title()+" No."+str(no+1), fontweight ="bold")
    plt.tight_layout()
    # いらないものを消す
    #fig.axes.get_xaxis().set_visible(False)
    #fig.axes.get_yaxis().set_visible(False)
    #plt.axis(False)
    # 画像を保存する
    plt.savefig(result_heatmaps_groundtruth_png_dir+tag+"-"+str(no)+".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()


def json_to_npy():

    nodule_info = []
    for v_b in visualize_sample_benign_numbers:
        nodule_info.append([v_b, 1])
    for v_m in visualize_sample_malignant_numbers:
        nodule_info.append([v_m, 0])

    names = np.array(nodule_info)[:,0]
    names = names.tolist()
    outputs = [[],[],[],[]]
    heatmaps = [[],[],[],[]]
    heatmap_aves = []

    for doctor_index in [1,2,3,4]:
        json_path_base = base_groundtruth_dir + str(doctor_index) +"/"

        for name in names:
            json_path = json_path_base + name + '.json'
            if(not os.path.isfile(json_path)):
                print("no file")
                outputs[doctor_index-1].append([3,""])
                heatmaps[doctor_index-1].append(np.zeros((48,48)))
                continue

            with open(json_path) as f:
                df = json.load(f)

            label_json = df["label"]
            marks = np.array(label_json["marks"])
            malignancy = int(label_json["malignancy"])
            comment = label_json["comment"]
            outputs[doctor_index-1].append([malignancy,comment])
            heatmaps[doctor_index-1].append(marks)

    with open(doctor_csv, 'w') as f:
        writer = csv.writer(f)
        for i in range(len(names)):
            writer.writerow([names[i], outputs[0][i][0], outputs[1][i][0], outputs[2][i][0], outputs[3][i][0], outputs[0][i][1], outputs[1][i][1], outputs[2][i][1], outputs[3][i][1]])
    for i in range(len(names)):
        hp = np.zeros((48,48))
        if nodule_info[i][1] == 0 and outputs[0][i][0] > 3:
            hp = hp + heatmaps[0][i]
        if nodule_info[i][1] == 0 and outputs[1][i][0] > 3:
            hp = hp + heatmaps[1][i]
        if nodule_info[i][1] == 0 and outputs[2][i][0] > 3:
            hp = hp + heatmaps[2][i]
        if nodule_info[i][1] == 0 and outputs[3][i][0] > 3:
            hp = hp + heatmaps[3][i]
        if nodule_info[i][1] == 1 and outputs[0][i][0] < 3:
            hp = hp + heatmaps[0][i]
        if nodule_info[i][1] == 1 and outputs[1][i][0] < 3:
            hp = hp + heatmaps[1][i]
        if nodule_info[i][1] == 1 and outputs[2][i][0] < 3:
            hp = hp + heatmaps[2][i]
        if nodule_info[i][1] == 1 and outputs[3][i][0] < 3:
            hp = hp + heatmaps[3][i]

        heatmap_aves.append(hp/4)

    images_b, names_b = get_sample_benigns()
    images_m, names_m = get_sample_malignants()

    heatmaps_b = []
    for name_b in names_b:
        j = names.index(os.path.splitext(os.path.basename(name_b))[0])
        heatmaps_b.append(heatmap_aves[j])
    heatmaps_b = np.array(heatmaps_b)
    save_heatmap(images_b, heatmaps_b, "benign")
    np.save(result_heatmaps_groundtruth_npy_dir+"b.npy",heatmaps_b)

    heatmaps_m = []
    for name_m in names_m:
        j = names.index(os.path.splitext(os.path.basename(name_m))[0])
        heatmaps_m.append(heatmap_aves[j])
    heatmaps_m = np.array(heatmaps_m)
    print(heatmaps_m.shape)
    save_heatmap(images_m, heatmaps_m, "malignant")
    np.save(result_heatmaps_groundtruth_npy_dir+"m.npy",heatmaps_m)
