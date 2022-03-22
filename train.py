"""
    学習用のプログラミング

    kerasを用いて学習を行う。
    10ホールドのクロス交差検証を行うことより、10回学習を行っている。

    データサイズが大きく、1エポックにすべてのデータを読み込むことが難しいので、
    オリジナルのデータジェネレーターを用いている。
    ./utils/train/generator.pyでコードを確認できる。
    
    また、アンバランスデータの解決のため、UnderSampling + Baggingを使用している。
    https://blog.amedama.jp/entry/under-bagging-kfold
"""

import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from functools import partial
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from keras.backend import clear_session
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from scipy import interp
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

import skimage
import numpy as np
import matplotlib.pyplot as plt

# 自作フォルダ
from utils.train.draw_graphs import *
from utils.train.metrics import *
from utils.train.under_bagging import UnderBaggingKFold
from networks import models
from networks.models import *
from settings import *

# 学習パラメーター
# batchサイズは2の階乗にするべし
epoch_num = 5
batch_num = 32
learning_rate = 5e-6

def train_model(experiment_index, experiment_name, data_argument):

    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    # For the functions where you wish to use GPUs, write something like the following:
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):

        # Run model with cross-validation and plot ROC curves
        cv = UnderBaggingKFold(n_splits=10)
        # cv = StratifiedKFold(n_splits=10)

        X, y = create_data()

        tprs = []
        aucs = []

        all_acc             = []
        all_loss            = []
        all_recall          = []
        all_precision       = []

        all_val_acc        = []
        all_val_loss       = []
        all_val_recall     = []
        all_val_precision  = []

        all_test_acc        = []
        all_test_recall     = []
        all_test_precision  = []
        all_test_f1_measure  = []

        model = single_size_network(experiment_index)
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=generate_metrics())
        model.save_weights(save_init_weight_dir+str(experiment_index)+".hdf5")
        plot_model(model, to_file=model_map_dir+experiment_name+'.png', show_shapes=True)

        for i, (train, test) in enumerate(cv.split(X, y)):

            model.load_weights(save_init_weight_dir+str(experiment_index)+".hdf5")
            print("|||||||||||||||||||||||| [ "+ experiment_name + " ] " + str(i)+" / 10 |||||||||||||||||||||")

            X_train1, Y_train = generate_train(X[train], y[train], data_argument)
            X_test1, Y_test = generate_test(X[test], y[test])
            Y_train_ca = to_categorical(Y_train)
            Y_test_ca = to_categorical(Y_test)

            gen_train = Generator()
            gen_test = Generator()
            gen_predict = Generator()

            print("train: "+str(len(X_train1))+" test: "+str(len(X_test1)))

            acc_test = []
            loss_test = []

            # EaelyStopping(早期打ち切り)の設定
            # val loss(検証用損失関数)が5エポック以上、下がらなければ終了
            early_stopping = EarlyStopping(
                            monitor='val_loss',
                            patience=5,
                            )


            history = model.fit_generator(generator=gen_train.flow_from_directory(X_train1, Y_train_ca, batch_num),
                epochs=epoch_num,
                steps_per_epoch=int(np.ceil(len(X_train1)/batch_num)),
                validation_data=gen_test.flow_from_directory(X_test1, Y_test_ca, batch_num), #callbacks=[lr_decay],
                validation_steps=int(np.ceil(len(X_test1)/batch_num)),
                max_queue_size = 1,
                callbacks = [early_stopping]
            )

            all_val_acc.append(history.history["val_acc"])
            all_val_loss.append(history.history["val_loss"])
            all_val_recall.append(history.history["val_recall"])
            all_val_precision.append(history.history["val_precision"])
            all_acc.append(history.history["acc"])
            all_loss.append(history.history["loss"])
            all_recall.append(history.history["recall"])
            all_precision.append(history.history["precision"])


            test_history = model.evaluate_generator(gen_predict.flow_from_directory(X_test1, Y_test_ca, 1), steps=len(X_test1))
            # print(model.metrics_names)
            # print(test_history)
            all_test_acc.append(test_history[1]) # acc
            all_test_recall.append(test_history[2]) # recall
            all_test_precision.append(test_history[3]) # precision
            all_test_f1_measure.append(test_history[4]) # precision

            y_pred = model.predict_generator(gen_predict.flow_from_directory(X_test1, Y_test_ca, 1), steps=len(X_test1))
            y_rate = []
            for y_p in y_pred:
                y_rate.append(y_p[0]/(y_p[0]+y_p[1]))

            fpr, tpr, threshold = roc_curve(Y_test_ca[:,0], y_rate)
            mean_fpr = np.linspace(0, 1, 100)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))

            # パラメータの保存
            model_json_str = model.to_json()
            open(save_model_dir+experiment_name+"_"+str(i)+'.json', 'w').write(model_json_str)
            model.save_weights(save_model_dir+experiment_name+"_"+str(i)+'.h5');

        # 結果を出力
        draw_accs(result_accs_dir + experiment_name,experiment_name, all_acc, all_val_acc, all_loss, all_val_loss)
        draw_roc(result_mets_dir + experiment_name, experiment_name, tprs, aucs)
        draw_metrics(result_roc_dir + experiment_name,experiment_name, all_recall, all_val_recall, all_precision, all_val_precision)
        draw_document(result_doc_dir + experiment_name, experiment_name, tprs, aucs, all_acc, all_recall, all_precision, all_test_f1_measure)


if __name__ =='__main__':

    import sys
    args = sys.argv
    train_model(int(args[1]), args[2], int(args[3]))
