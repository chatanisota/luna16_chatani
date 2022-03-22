"""
    生成されたヒートマップを、見やすいように
    レイヤー対深層学習モデルで、
    一枚の画像に並べるプログラム

    https://note.com/kuronekohomuhomu/n/n314d2f92e6a5
"""
import matplotlib.pyplot as plt
import os
import shutil
import tqdm
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from utils.visualize.samples import *
from settings import *


def renew_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)


def print_heatmaps(img_names, b_or_m = "benign"):
    networks = [
        [1, "CNN", 2],
        [4, "SECNN", 4],
        [6, "CBAM-CNN", 6],
        [2, "CNNResnet", 6],
        [3, "SECNNResnet", 8],
        [5, "CBAM-CNNResnet", 10]
    ]

    print("runnning...")
    #for dataset, experiments_sets in zip(['D1','D2'],[experiments[:3], experiments[3:]]):
    for img_name in tqdm.tqdm(img_names):
        img_name = os.path.splitext(os.path.basename(img_name))[0]
        for dataset_index in [0,1,2]:
            plt.clf()
            plt.title("no."+str(img_name)+" "+b_or_m)
            fig, axes = plt.subplots(10,6,figsize=(20,16), constrained_layout=True)
            plt.axis(False)
            for col,network in enumerate(networks):
                #id = img_name + "-" + str(network_index+1) + "-" + str(dataset_index)+ "-9-"+b_or_m[0]
                #img = plt.imread("./heatmaps/outputs/"+id+".png")
                #axes[0, network_index].set_axis_off()
                #axes[0, network_index].imshow(img)
                for conv_index in range(10):
                    if(conv_index >= network[2]):
                        axes[conv_index, col].set_axis_off()
                        continue
                    #axes[conv_index, 0].set_ylabel("conv2d_"+str(conv_index))
                    if(conv_index==1):
                        axes[0, col].set_title(network[1])

                    id = img_name + "-" + str(network[0]) + "-" + str(dataset_index)+ "-"+b_or_m[0]+"-"+str(conv_index)
                    img = plt.imread(result_heatmaps_png_dir+id+".png")
                    #img = plt.imread("./outputs/"+id+".png")
                    axes[conv_index, col].set_axis_off()
                    axes[conv_index, col].imshow(img)
            img = plt.imread(result_heatmaps_png_dir+id+".png")
            plt.savefig(result_heatmaps_experiments_vs_layers_dir+"D"+str(dataset_index+1)+"_"+b_or_m+"_"+str(img_name)+".png")
            plt.close()


def paste_hetmaps():
    files = glob.glob(result_heatmaps_experiments_vs_layers_dir+"*")
    for file in tqdm.tqdm(files):
        codes = os.path.basename(file).split('_')
        image_name_and_png = codes[-3] +'_'+ codes[-2] + '_' + codes[-1]
        imn = Image.open(result_heatmaps_png_dir+image_name_and_png)
        b_or_m = 'benign' if os.path.isfile(result_heatmaps_experiments_vs_layers_dir+"D1_benign_"+image_name_and_png) else 'malignant'
        for dataset_index in [0,1,2]:
            im1 = Image.open(result_heatmaps_experiments_vs_layers_dir+"D"+str(dataset_index+1)+"_"+b_or_m+"_"+image_name_and_png)
            im1.paste(imn, (0, im1.height - imn.height))
            im1.save(result_heatmaps_experiments_vs_layers_dir+"D"+str(dataset_index+1)+"_"+b_or_m+"_"+image_name_and_png)




from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
renew_folder(result_heatmaps_experiments_vs_layers_dir)
_, b_list = get_sample_benigns()
print_heatmaps(b_list, "benign")
_, m_list = get_sample_malignants()
print_heatmaps(m_list, "malignant")
paste_hetmaps()
