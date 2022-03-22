# -*- coding:utf-8 -*-

'''
学習・可視化はこのプログラムを実行してください。
前処理は別途「preprocessing.py」を実行してください。
'''
import os
import subprocess

# model番号 実験名 データ拡張(0:なし, 1:回転, 2:回転+垂直反転)
experiments = [
    [1, "CNN2D(D1)", 0],
    [2, "CNNResnet2D(D1)", 0],
    [3, "SECNNResnet2D(D1)",0],
    [4, "SECNN2D(D1)",0],
    [5, "CBAM-CNNResnet2D(D1)",0],
    [6, "CBAM-CNN2D(D1)",0],

    [1, "CNN2D(D2)",1],
    [2, "CNNResnet2D(D2)", 1],
    [3, "SECNNResnet2D(D2)",1],
    [4, "SECNN2D(D2)",1],
    [5, "CBAM-CNNResnet2D(D2)",1],
    [6, "CBAM-CNN2D(D2)",1],

    [1, "CNN2D(D3)",2],
    [2, "CNNResnet2D(D3)", 2],
    [3, "SECNNResnet2D(D3)",2],
    [4, "SECNN2D(D3)",2],
    [5, "CBAM-CNNResnet2D(D3)",2],
    [6, "CBAM-CNN2D(D3)",2],
]




def train():
    print(" beigin train...")
    for i,ex in enumerate(experiments):
        print("_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/")
        print("_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/")
        print("_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/")
        print("_/_/_/_/_/["+str(i)+"] "+ex[1]+"_/_/_/_/_/_/_/_/_/_/_/_/_/")
        print("_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/")
        print("_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/")

        #if(ex[1]==1):
            #cmd = "python easy_model.py"+" "+str(ex[0])+" "+str(ex[2])
        #else:
        cmd = "python train.py"+" "+str(ex[0])+" "+str(ex[1])+" "+str(ex[2])
        print("COMMAND: ",cmd)
        runcmd = subprocess.call(cmd.split())
        print (runcmd)

def visualize():
    print(" beigin visualize...")

    """
    for i,ex in enumerate(experiments):
        print("_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/")
        print("_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/")
        print("_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/")
        print("_/_/_/_/_/["+str(i)+"] "+ex[1]+"_/_/_/_/_/_/_/_/_/_/_/_/_/")
        print("_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/")
        print("_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/")

        #if(ex[1]==1):
            #cmd = "python easy_model.py"+" "+str(ex[0])+" "+str(ex[2])
        #else:
        cmd = "python visualize.py"+" "+str(ex[0])+" "+str(ex[1])+" "+str(ex[2])
        print("COMMAND: ",cmd)
        runcmd = subprocess.call(cmd.split())
        print (runcmd)
    """
    print(" begin arrange...")
    cmd = "python arrange_heatmaps.py"
    print("COMMAND: ",cmd)
    runcmd = subprocess.call(cmd.split())
    print (runcmd)

if __name__ =='__main__':

    import sys
    args = sys.argv
    if(len(args) != 2):
        print("train/visualize")

    if(args[1]=='train'):
        train()
    elif(args[1]=='visualize'):
        visualize()
    else:
        print("train/visualize")
