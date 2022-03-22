"""
    ヒートマップを描画するライブラリ
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable
from settings import *

def standarization(n_arrays):
    for i in range(len(n_arrays)):
        if(n_arrays[i].max()>0):
            n_arrays[i] = n_arrays[i] / n_arrays[i].max()
    return n_arrays

def renew_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)


def draw_heatmap(imgs, names, heatmaps, id, target_layer=None):
    for i in range(heatmaps.shape[0]):
        inner_draw_heatmap(i, imgs[i], names[i], heatmaps[i], id, target_layer)


def inner_draw_heatmap(no, img, img_name, heatmap, id, target_layer=None):
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

    fig.subplots_adjust(wspace=0.1)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if colorbar:
        fig.colorbar(im2, cax=cax)


    plt.tight_layout(rect=[0,0,1,0.96])
    if(target_layer != None):
        title_name = target_layer.replace('_', " ").title()
        fig.suptitle(title_name, fontsize=19)#fontweight ="bold"
    # いらないものを消す
    #fig.axes.get_xaxis().set_visible(False)
    #fig.axes.get_yaxis().set_visible(False)
    #plt.axis(False)
    # 画像を保存する
    plt.savefig(result_heatmaps_png_dir + os.path.splitext(os.path.basename(img_name))[0]+"-"+id+".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

def draw_image(imgs, names, b_or_m=""):
    for i, img in enumerate(imgs):
        plt.clf()
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot()
        plt.title(b_or_m+" No."+str(i+1))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.imshow(img, cmap="gray", vmax=1.0, vmin=0.0)
        plt.savefig(result_heatmaps_png_dir + os.path.splitext(os.path.basename(names[i]))[0]+".png" ,bbox_inches='tight', pad_inches=0.1)
        plt.close()
