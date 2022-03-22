"""
    学習性能指標であるROCやACCのグラフを描写するライブラリ
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
import math

colors = [
    "orange",
    "pink",
    "yellow",
    "lime",
    "green",
    "cyan",
    "blue",
    "purple",
    "gray",
    "black",
]

def calc_view_epoch(lists):
    max_epoch = max([len(l) for l in lists])
    view_epoch = math.ceil(max_epoch/50)
    return view_epoch * 50

# ROCを生成
def draw_roc(save_roc_path, experiment_name, tprs, aucs):

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=experiment_name+" Receiver operating characteristic")
    ax.legend(loc="lower right")
    plt.savefig(save_roc_path)


# ACCの学習推移結果を生成
def draw_accs(save_path, experiment_name, all_acc, all_test_acc, all_loss, all_test_loss):

    view_epoch = calc_view_epoch(all_acc)
    # Plot the loss in the history
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
    for i in range(10):
        ax1.plot(all_acc[i],        label="train (cv"+str(i)+")", color=colors[i], linestyle="dotted")
        ax1.plot(all_test_acc[i],   label="test  (cv"+str(i)+")", color=colors[i])
    #ax1.plot(np.mean(np.array(all_acc), axis=0),        label="train mean", color="red", linewidth=3, linestyle="dotted")
    #ax1.plot(np.mean(np.array(all_test_acc), axis=0),        label="train mean", color="red", linewidth=3)
    ax1.set_title(experiment_name+' Accuracy')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.set_xlim(0, view_epoch)
    ax1.set_ylim(0.5, 1.0)
    #ax1.legend(loc='lower right', fontsize=5)

    for i in range(10):
        ax2.plot(all_loss[i],        label="train (cv"+str(i)+")", color=colors[i], linestyle="dotted")
        ax2.plot(all_test_loss[i],   label="test  (cv"+str(i)+")", color=colors[i])
    #ax2.plot(np.mean(np.array(all_loss), axis=0),        label="train mean", color="red", linewidth=3, linestyle="dotted")
    #ax2.plot(np.mean(np.array(all_test_loss), axis=0),        label="train mean", color="red", linewidth=3)
    ax2.set_title(experiment_name+' Loss')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_xlim(0, view_epoch)
    ax2.set_ylim(0.0, 0.8)
    #ax2.legend(loc='upper right', fontsize=5)
    ax2.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)


# 評価指標の学習推移グラフを生成
def draw_metrics(save_path, experiment_name, all_recall, all_test_recall, all_precision, all_test_precision):

    view_epoch = calc_view_epoch(all_recall)

    plt.clf()
    fig, (ax3, ax4) = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
    for i in range(10):
        ax3.plot(all_recall[i],        label="train (cv"+str(i)+")", color=colors[i], linestyle="dotted")
        ax3.plot(all_test_recall[i],   label="test  (cv"+str(i)+")", color=colors[i])
    #ax3.plot(np.mean(np.array(all_recall), axis=0),        label="train mean", color="red", linewidth=3, linestyle="dotted")
    #ax3.plot(np.mean(np.array(all_test_recall), axis=0),        label="train mean", color="red", linewidth=3)
    ax3.set_title(experiment_name+' Recall')
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('recall')
    ax3.set_xlim(0, view_epoch)
    ax3.set_ylim(0.5, 1.0)
    #ax3.legend(loc='lower right', fontsize=5)

    for i in range(10):
        ax4.plot(all_precision[i],        label="train (cv"+str(i)+")", color=colors[i], linestyle="dotted")
        ax4.plot(all_test_precision[i],   label="test  (cv"+str(i)+")", color=colors[i])
    #ax4.plot(np.mean(np.array(all_precision), axis=0),        label="train mean", color="red", linewidth=3, linestyle="dotted")
    #ax4.plot(np.mean(np.array(all_test_precision), axis=0),        label="train mean", color="red", linewidth=3)
    ax4.set_title(experiment_name+' Precision')
    ax4.set_xlabel('epoch')
    ax4.set_ylabel('precision')
    ax4.set_xlim(0, view_epoch)
    ax4.set_ylim(0.5, 1.0)
    ax4.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')
    #ax4.legend(loc='lower right', fontsize=5)
    plt.tight_layout()
    plt.savefig(save_path)

# 詳しい評価指標の結果を書いたテキストファイルを作成
def draw_document(path, experiment_name, tprs, aucs, all_test_acc, all_test_recall, all_test_precision, all_test_f1_measure):

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    f = open(path+'.txt', mode='wt', encoding='utf-8')
    f.write(experiment_name+'\n')
    f.write('auc:' + '%0.3f +/- %0.3f' % (mean_auc, std_auc) + '\n')
    f.write('acc: ' + '{:.03f}'.format(np.mean([np.mean(a) for a in all_test_acc]))+'\n')
    f.write('recall: ' + '{:.03f}'.format(np.mean([np.mean(a) for a in all_test_recall]))+'\n')
    f.write('precision: ' + '{:.03f}'.format(np.mean([np.mean(a) for a in all_test_precision]))+'\n')
    f.write('f1_mesure: ' + '{:.03f}'.format(np.mean([np.mean(a) for a in all_test_f1_measure]))+'\n')
    f.close()
