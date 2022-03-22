"""
    可視化用プログラム

    打田研究室のGitHubを利用した。
    https://github.com/uchidalab/softmaxgradient-lrp
    打田氏のソースコードには、iNNvestigateが使用されている。
    https://github.com/albermax/innvestigate

    iNNvestigateは元々、SmoothGradやGradCAMなど、深層学習を可視化する関数が用意されており、
    打田氏はiNNvestigateを内包し、LRPやGuidedGradCAMを利用しやすくした。
    しかしながら、GitHubのソースコードではうまく動かないことが
    多々あったので、iNNvestigateや打田氏のコードを動くようになるまで鬼改造を施した。
    （もう２度とこのソースコードを再現できない気がする。）

    ＜＜＜重要＞＞＞
    したがって、
    iNNvestigateはすでにこのプロジェクトに追加しており、
    iNNvestigateをinstallする必要はない。
    ライブラリのバージョンが異なると動かなくなることが多々あるため、
    実験環境.txtに必須バージョン情報記載している。

    # h5あたりのバグが出たら……
    https://qiita.com/ankomotch/items/52baffedf43ffb2385d8
    https://pypi.org/project/innvestigate/

    今回はGuided Grad CAMで行っているが、
    GradCAM, GuidedGradCAM, GBP, LRP, CLRP, LRPA, LRPB
    などに対応することができるため、
    ソースコードを適切に編集しながら、試してもらってもかまわない。
"""
from keras.models import load_model, Model
from keras.models import model_from_json
from tensorflow.python.keras.utils.vis_utils import plot_model
from keras.preprocessing.image import img_to_array, load_img
from utils.visualize.uchida_vis import GradCAM, GuidedGradCAM, GBP, LRP, CLRP, LRPA, LRPB
from utils.innvestigate import utils as iutils
import os
import glob
import tqdm
import matplotlib.pyplot as plt
import numpy as np

from utils.visualize.samples import *
from utils.visualize.draw_heatmaps import *

def main(network_index, experiment_name, data_argument):
    # limits tensorflow to a specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    images_b, names_b = get_sample_benigns()
    images_m, names_m = get_sample_malignants()

    draw_image(images_b, names_b, "Benign")
    draw_image(images_m, names_m, "Malignant")

    id = str(network_index)+'-'+str(data_argument)
    target_layers = []

    # 可視化する層一覧
    # ブロックの前後を可視化するように、可視化箇所を設定した
    # 詳しい選定理由については論文参照
    if(network_index==1):
        #CNN
        target_layers.append("max_pooling2d_1")
        target_layers.append("max_pooling2d_2")
    elif(network_index==2):
        #CNNResnet
        target_layers.append("max_pooling2d_1")
        target_layers.append("batch_normalization_2")
        target_layers.append("res1_add")
        target_layers.append("max_pooling2d_2")
        target_layers.append("batch_normalization_4")
        target_layers.append("res2_add")
    elif(network_index==3):
        #SECNNResnet
        target_layers.append("max_pooling2d_1")
        target_layers.append("batch_normalization_2")
        target_layers.append("res1_multiply")
        target_layers.append("res1_add")
        target_layers.append("max_pooling2d_2")
        target_layers.append("batch_normalization_4")
        target_layers.append("res2_multiply")
        target_layers.append("res2_add")
    elif(network_index==4):
        #SECNN
        target_layers.append("max_pooling2d_1")
        target_layers.append("se1_multiply")
        target_layers.append("max_pooling2d_2")
        target_layers.append("se2_multiply")
    elif(network_index==5):
        #CBAM-CNNResnet
        target_layers.append("max_pooling2d_1")
        target_layers.append("batch_normalization_2")
        target_layers.append("res1_attention_ch_mul1")
        target_layers.append("res1_attention_sp_mul1")
        target_layers.append("res1_add")
        target_layers.append("max_pooling2d_2")
        target_layers.append("batch_normalization_4")
        target_layers.append("res2_attention_ch_mul1")
        target_layers.append("res2_attention_sp_mul1")
        target_layers.append("res2_add")
    elif(network_index==6):
        #CBAM-CNNResnet
        target_layers.append("max_pooling2d_1")
        target_layers.append("cbam1_attention_ch_mul1")
        target_layers.append("cbam1_attention_sp_mul1")
        target_layers.append("max_pooling2d_2")
        target_layers.append("cbam2_attention_ch_mul1")
        target_layers.append("cbam2_attention_sp_mul1")

    print("processing... please wait.")

    for cv_index in tqdm.tqdm(range(10)):
    # モデルを読み込む
        model = model_from_json(open(save_model_dir+experiment_name+'_'+str(cv_index)+'.json').read())

        # 学習結果を読み込む
        model.load_weights(save_model_dir+experiment_name+'_'+str(cv_index)+'.h5')

        partial_model = Model(
            inputs=model.inputs,
            outputs=iutils.keras.graph.pre_sigmoid_tensors(model.outputs),
            name=model.name,
        )

        # 一度可視化して、cvごとのヒートマップデータを保存
        predicts_b = model.predict(images_b)
        predicts_m = model.predict(images_m)
        inner_id = id + "-" + str(cv_index) + "-b"
        visualize(partial_model, target_layers, inner_id, images_b, names_b, predicts_b[:,1], 1)
        inner_id = id + "-" + str(cv_index) + "-m"
        visualize(partial_model, target_layers, inner_id, images_m, names_m, predicts_m[:,0], 0)

    # cvを平均化して、ヒートマップを描画
    heatmap_all = None
    for b_or_m in ['b', 'm']:
        for i, target_layer in enumerate(tqdm.tqdm(target_layers)):

            heatmap_result = None
            for cv_index in range(10):
                inner_id = id + "-" + str(cv_index) + "-" + b_or_m
                heatmaps = np.load(result_heatmaps_npy_dir + inner_id+'-'+str(i)+'.npy')
                if(cv_index==0):
                    heatmap_result = heatmaps
                else:
                    heatmap_result = heatmap_result + heatmaps
            heatmap_result = standarization(heatmap_result)
            if i == 0:
                heatmap_all = heatmap_result
            else:
                heatmap_all = heatmap_result + heatmap_all

            if b_or_m == 'b':
                draw_heatmap(images_b, names_b, heatmap_result, id+"-"+b_or_m+'-'+str(i), target_layer)
            else:
                draw_heatmap(images_m, names_m, heatmap_result, id+"-"+b_or_m+'-'+str(i), target_layer)

    # CAM系以外を試したい場合はこっちを使う
    """
    heatmap_all = None
    # cvを平均化して、ヒートマップを描画
    for b_or_m in ['b', 'm']:
        heatmap_result = None
        for cv_index in range(10):
            inner_id = id + "-" + str(cv_index) + "-" + b_or_m
            heatmaps = np.load(result_heatmaps_npy_dir + inner_id+'.npy')
            if(cv_index==0):
                heatmap_result = heatmaps
            else:
                heatmap_result = heatmap_result + heatmaps
        heatmap_result = standarization(heatmap_result)
        if i == 0:
            heatmap_all = heatmap_result
        else:
            heatmap_all = heatmap_result + heatmap_all

        if b_or_m == 'b':
            draw_heatmap(images_b, names_b, heatmap_result, id+"-"+b_or_m, target_layer)
        else:
            draw_heatmap(images_m, names_m, heatmap_result, id+"-"+b_or_m, target_layer)
    """



def visualize(partial_model, target_layers, inner_id, input_imgs, input_names, predicts, target_class):
    # target_class = 1 # 良性

    #
    # GuidedGradCAM
    #

    for t,target_layer in enumerate(target_layers):
        guidedgradcam_analyzer = GuidedGradCAM(
            partial_model,
            target_id=target_class,
            layer_name=target_layer,
            relu=False,
        )
        heatmaps = guidedgradcam_analyzer.analyze(input_imgs)
        heatmaps = heatmaps.sum(axis=(3))
        heatmaps = standarization(heatmaps)
        np.save(result_heatmaps_npy_dir+inner_id+'-'+str(t)+'.npy', heatmaps)

    # 他の可視化を試したい場合は下をどうぞ、ただし、Grad CAM以外はレイヤーごとに可視化できないので注意
    # 動作確認してません
    """
    #
    # Grad CAM
    #

    for t,target_layer in enumerate(target_layers):
        gradcam_analyzer = GradCAM(
            model=partial_model,
            target_id=target_class,
            layer_name=target_layer,
            relu=True,
        )
        heatmaps = partial_gradcam_analyzer.analyze(input_imgs)
        heatmaps = analysis_partial_grad_cam.sum(axis=(3))
        np.save(result_heatmaps_npy_dir+inner_id+'-'+str(t)+'.npy', heatmaps)


    #
    # Guided Back Propagation
    #

    guidedbackprop_analyzer = GBP(
        partial_model,
        target_id=target_class,
        relu=True,
    )
    heatmaps = guidedbackprop_analyzer.analyze(input_imgs)
    np.save(result_heatmaps_npy_dir+inner_id+'.npy', heatmaps)

    #
    # LRP
    #

    lrp_analyzer = LRP(
        partial_model,
        target_id=target_class,
        relu=True,
        low=0.0,
        high=1.0,
    )
    heatmaps = lrp_analyzer.analyze(input_imgs)
    np.save(result_heatmaps_npy_dir+inner_id+'.npy', heatmaps)

    #
    # CLRP
    #

    clrp_analyzer = CLRP(
        partial_model,
        target_id=target_class,
        relu=True,
        low=0.0,
        high=1.0,
    )
    heatmaps = clrp_analyzer.analyze(input_imgs)
    np.save(result_heatmaps_npy_dir+inner_id+'.npy', heatmaps)

    #
    # SGLRP
    #

    sglrp_analyzer = SGLRP(
        partial_model,
        target_id=target_class,
        relu=True,
        low=0.0,
        high=1.0,
    )
    heatmaps = sglrp_analyzer.analyze(input_imgs)
    np.save(result_heatmaps_npy_dir+inner_id+'.npy', heatmaps)
    """


if __name__ =='__main__':

    import sys
    args = sys.argv
    main(int(args[1]), args[2], int(args[3]))
