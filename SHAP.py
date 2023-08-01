import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score, balanced_accuracy_score
from keras.optimizers import Adam
from sklearn.model_selection import KFold, GroupKFold, train_test_split, StratifiedKFold
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from imblearn.under_sampling import RandomUnderSampler
from keras.models import load_model
import sys
import shap
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# import data_load
# import feature_code
import network
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.models import Sequential
from keras.models import load_model
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# from tensorflow.compat.v1.keras.backend import get_session

tf.compat.v1.disable_v2_behavior()


def noramlization(data, filename):
    f = open(filename, 'r')
    sequence = []
    num = []
    print(data.shape)
    for line in f.readlines():
        if line[0] != ' ':
            if line[0] != '>':
                seq = line.upper().strip('\n')
                sequence.append(line.upper().strip('\n'))
                num.append(len(seq))
    num = np.array(num).reshape(-1, 1)
    num_nor = data / num
    return num_nor


def binary_focal_loss(alpha, gamma):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


# data load
cell_line = 'NHLF'
file_train = 'data/train/' + cell_line + '.fasta'
x_train = np.loadtxt('feature/motifcount/pvalue0.0001/' + cell_line + '_train_motif.txt')
x_train = noramlization(x_train, file_train)

file_test = 'data/test/' + cell_line + '.fasta'
x_test = np.loadtxt('feature/motifcount/pvalue0.0001/' + cell_line + '_test_motif.txt')
x_test = noramlization(x_test, file_test)

y_train = np.loadtxt('data/train/' + cell_line + '_y_train.txt')
y_test = 'data/test/' + cell_line + '_y_test.txt'

data_shape = x_train.shape[1:2]

# parameter setting
MAX_EPOCH = 50
BATCH_SIZE = 100
learning_rate = 0.0001
dropout_rate1 = 0.6
alpha = 0.5
gamma = 3
# model construction

LOSS = binary_focal_loss(alpha, gamma)
model = network.linear_motif(data_shape, dropout_rate1, )

# filepath = "model/ceshi/" + cell_line + "_model_cnn.hdf5"
# filepath = "model/dna2vecdim500/" + cell_lines + "_" + str(kernel_num) + "_" + str(
#     kernel_size) + "_" + str(
#     pool_size) + "_" + str(
#     dropout_rate1) + "_" + str(dropout_rate2) + "_" + str(stride) + "_model_cnn.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10)
model.compile(loss=LOSS, optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
kf = StratifiedKFold(5, shuffle=True, random_state=10).split(x_train, y_train)
for j, (train_index, test_index) in enumerate(kf):
    history = model.fit(x_train[train_index],
                        y=y_train[train_index],
                        batch_size=BATCH_SIZE, epochs=MAX_EPOCH,
                        validation_data=(
                            x_train[test_index],
                            y_train[test_index]),
                        callbacks=[early_stopping_monitor])
    print(j + 1)

# early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=5)
# model = network.linear_motif(data_shape,dropout_rate1)
# model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
# model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCH, validation_data=(
#     x_val, y_val), callbacks=[early_stopping_monitor])
# SHAP analysis
shap.initjs()
# background = x_train[np.random.choice(x_train.shape[0], 2000, replace=False)]
explainer = shap.DeepExplainer(model, x_train)
shap_values = explainer.shap_values(x_test)
# print(np.array(shap_values).shape)
shap_values = np.squeeze(np.array(shap_values))
feature_importance = np.abs(shap_values).mean(axis=0)
# 获取特征贡献度从大到小的排序索引
sorted_indices = np.argsort(feature_importance)[::-1]
# 输出特征索引
print(np.array(sorted_indices))
# 输出shap图
shap.summary_plot(shap_values, x_test, max_display=50)
print(cell_line)
# shap.summary_plot(shap_values, x_test, plot_type="bar")
