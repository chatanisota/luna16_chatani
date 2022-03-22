"""
    医師の診断と、可視化画像を比較するためのプログラム

    dice,jaccard,simpsonそれぞれのIoUが得られる。
    groundtruthフォルダには、mark_meソフトウェアで取得されたjsonファイルをいれておくこと。
    
"""
import numpy as np
import cv2
import csv
from settings import *
from utils.groundtruth.json_to_npy import *

HEATMAP_DOCTOR_M = result_heatmaps_groundtruth_npy_dir + "m.npy"
HEATMAP_DOCTOR_B = result_heatmaps_groundtruth_npy_dir + "b.npy"

if not os.path.isfile(HEATMAP_DOCTOR_M):
    # フォルダがないとき
    json_to_npy()

if not os.path.isfile(HEATMAP_DOCTOR_M):
    print("error: groundtruthのjsonから上手くheatmapのnpyが作れませんでした。")
    exit()

heatmap_doctor_m = np.load(HEATMAP_DOCTOR_M)
ground_truths_m = heatmap_doctor_m>0
heatmap_doctor_b = np.load(HEATMAP_DOCTOR_B)
ground_truths_b = heatmap_doctor_b>0

networks = [
    [1, "CNN", 2],
    [4, "SECNN", 4],
    [6, "CBAM-CNN", 6],
    [2, "CNNResnet", 6],
    [3, "SECNNResnet", 8],
    [5, "CBAM-CNNResnet", 10]
]

# 良性
f1 = open(result_compare_jaccard_b_csv, 'w')
f1.write('dataset, network, layer, 1,2,3,4,5,6,7,8,9\n')
f2 = open(result_compare_dice_b_csv, 'w')
f2.write('dataset, network, layer, 1,2,3,4,5,6,7,8,9\n')
f3 = open(result_compare_simpson_b_csv, 'w')
f3.write('dataset, network, layer, 1,2,3,4,5,6,7,8,9\n')
for dataset in [0,1,2]:
    for network in networks:
        for layer_id in range(network[2]):
            jaccard_all = []
            dice_all = []
            simpson_all = []
            for cv in range(10):
                id = result_heatmaps_npy_dir+str(network[0])+"-"+str(dataset)+"-"+str(cv)+"-b-"+str(layer_id)+".npy"
                heatmap_convs = np.load(id)
                active_map = np.array(np.zeros(heatmap_convs.shape),dtype=bool)
                for m in range(heatmap_convs.shape[0]):
                    active_map[m,:,:] = heatmap_convs[m,:,:] > heatmap_convs[m,:,:].mean()
                #active_map = heatmap_convs > 0.5
                #print(np.count_nonzero(active_map))
                jaccard = []
                dice = []
                simpson = []
                for i in range(9):
                    if np.count_nonzero(ground_truths_b[i] | active_map[i]) == 0:
                        jaccard.append(0)
                    else:
                        jaccard.append(np.count_nonzero(ground_truths_b[i] & active_map[i]) / np.count_nonzero(ground_truths_b[i] | active_map[i]))

                    if (np.count_nonzero(ground_truths_b[i])+np.count_nonzero(active_map[i])) == 0:
                        dice.append(0)
                    else:
                        dice.append(2 * np.count_nonzero(ground_truths_b[i] & active_map[i]) / (np.count_nonzero(ground_truths_b[i])+np.count_nonzero(active_map[i])))

                    if min(np.count_nonzero(ground_truths_b[i]),np.count_nonzero(active_map[i])) == 0:
                        simpson.append(0)
                    else:
                        simpson.append(np.count_nonzero(ground_truths_b[i] & active_map[i]) / min(np.count_nonzero(ground_truths_b[i]),np.count_nonzero(active_map[i])))

                jaccard_all.append(jaccard)
                dice_all.append(dice)
                simpson_all.append(simpson)
            jaccard_res = np.array(jaccard_all).mean(axis=0)
            dice_res = np.array(dice_all).mean(axis=0)
            simpson_res = np.array(simpson_all).mean(axis=0)
            csv_text = 'D'+str(dataset+1)+', '+network[1]+', '+str(layer_id)
            for i in range(9):
                csv_text = csv_text + ', '+"{:.5f}".format(jaccard_res[i])
            csv_text = csv_text + '\n'
            f1.write(csv_text)
            csv_text = 'D'+str(dataset+1)+', '+network[1]+', '+str(layer_id)
            for i in range(9):
                csv_text = csv_text + ', '+"{:.5f}".format(dice_res[i])
            csv_text = csv_text + '\n'
            f2.write(csv_text)
            csv_text = 'D'+str(dataset+1)+', '+network[1]+', '+str(layer_id)
            for i in range(9):
                csv_text = csv_text + ', '+"{:.5f}".format(simpson_res[i])
            csv_text = csv_text + '\n'
            f3.write(csv_text)
f1.close()
f2.close()
f3.close()

# 悪性
f1 = open(result_compare_jaccard_m_csv, 'w')
f1.write('dataset, network, layer, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n')
f2 = open(result_compare_dice_m_csv, 'w')
f2.write('dataset, network, layer, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n')
f3 = open(result_compare_simpson_m_csv, 'w')
f3.write('dataset, network, layer, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\n')
for dataset in [0,1,2]:
    for network in networks:
        for layer_id in range(network[2]):
            jaccard_all = []
            dice_all = []
            simpson_all = []
            for cv in range(10):
                id = result_heatmaps_npy_dir+str(network[0])+"-"+str(dataset)+"-"+str(cv)+"-m-"+str(layer_id)+".npy"
                heatmap_convs = np.load(id)
                active_map = np.array(np.zeros(heatmap_convs.shape),dtype=bool)
                for m in range(heatmap_convs.shape[0]):
                    active_map[m,:,:] = heatmap_convs[m,:,:] > heatmap_convs[m,:,:].mean()
                #active_map = heatmap_convs > 0.5
                #print(np.count_nonzero(active_map))
                jaccard = []
                dice = []
                simpson = []
                for i in range(15):
                    if np.count_nonzero(ground_truths_m[i] | active_map[i]) == 0:
                        jaccard.append(0)
                    else:
                        jaccard.append(np.count_nonzero(ground_truths_m[i] & active_map[i]) / np.count_nonzero(ground_truths_m[i] | active_map[i]))

                    if (np.count_nonzero(ground_truths_m[i])+np.count_nonzero(active_map[i])) == 0:
                        dice.append(0)
                    else:
                        dice.append(2 * np.count_nonzero(ground_truths_m[i] & active_map[i]) / (np.count_nonzero(ground_truths_m[i])+np.count_nonzero(active_map[i])))

                    if min(np.count_nonzero(ground_truths_m[i]),np.count_nonzero(active_map[i])) == 0:
                        simpson.append(0)
                    else:
                        simpson.append(np.count_nonzero(ground_truths_m[i] & active_map[i]) / min(np.count_nonzero(ground_truths_m[i]),np.count_nonzero(active_map[i])))

                jaccard_all.append(jaccard)
                dice_all.append(dice)
                simpson_all.append(simpson)
            jaccard_res = np.array(jaccard_all).mean(axis=0)
            dice_res = np.array(dice_all).mean(axis=0)
            simpson_res = np.array(simpson_all).mean(axis=0)
            csv_text = 'D'+str(dataset+1)+', '+network[1]+', '+str(layer_id)
            for i in range(15):
                csv_text = csv_text + ', '+"{:.5f}".format(jaccard_res[i])
            csv_text = csv_text + '\n'
            f1.write(csv_text)
            csv_text = 'D'+str(dataset+1)+', '+network[1]+', '+str(layer_id)
            for i in range(15):
                csv_text = csv_text + ', '+"{:.5f}".format(dice_res[i])
            csv_text = csv_text + '\n'
            f2.write(csv_text)
            csv_text = 'D'+str(dataset+1)+', '+network[1]+', '+str(layer_id)
            for i in range(15):
                csv_text = csv_text + ', '+"{:.5f}".format(simpson_res[i])
            csv_text = csv_text + '\n'
            f3.write(csv_text)
f1.close()
f2.close()
f3.close()
