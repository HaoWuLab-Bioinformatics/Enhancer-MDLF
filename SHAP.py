import argparse

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import KFold, GroupKFold, train_test_split, StratifiedKFold
import tensorflow as tf
from tensorflow.keras import backend as K
from imblearn.under_sampling import RandomUnderSampler
import sys
import shap
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# import data_load
# import feature_code
import network
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# from tensorflow.compat.v1.keras.backend import get_session

tf.compat.v1.disable_v2_behavior()
parser = argparse.ArgumentParser(description='SHAP.')
parser.add_argument('--lr',default=0.0001,help='learning rate')
parser.add_argument('--max_epoch',default=200,help='max_epoch for training')
parser.add_argument('--batch_size',default=100,help='max_epoch for training')
parser.add_argument('--dropout_rate1',default=0.6,help='dropout_rate for model before concatenate')
parser.add_argument('--alpha',default=0.5,help='alpha for focal loss')
parser.add_argument('--gamma',default=3,help='gamma for focal loss')
parser.add_argument('--cell_line',default='GM12878',help='cell line for train and prediction')


args = parser.parse_args()

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
cell_line = args.cell_line
file_train = 'data/train/' + cell_line + '.fasta'
x_train = np.loadtxt('feature/motif/' + cell_line + '_train_motif.txt')
x_train = noramlization(x_train, file_train)

file_test = 'data/test/' + cell_line + '.fasta'
x_test = np.loadtxt('feature/motif/' + cell_line + '_test_motif.txt')
x_test = noramlization(x_test, file_test)

y_train = np.loadtxt('data/train/' + cell_line + '_y_train.txt')
y_test = 'data/test/' + cell_line + '_y_test.txt'

data_shape = x_train.shape[1:2]

# parameter setting
MAX_EPOCH = args.max_epoch
BATCH_SIZE = args.batch_size
learning_rate = args.lr
dropout_rate1 = args.dropout_rate1
alpha = args.alpha
gamma = args.gamma
# model construction

LOSS = binary_focal_loss(alpha, gamma)
model = network.linear_motif(data_shape, dropout_rate1, )

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

# SHAP analysis
shap.initjs()
explainer = shap.DeepExplainer(model, x_train)
shap_values = explainer.shap_values(x_test)

shap_values = np.squeeze(np.array(shap_values))
feature_importance = np.abs(shap_values).mean(axis=0)

sorted_indices = np.argsort(feature_importance)[::-1]

print(np.array(sorted_indices))

y_base = explainer.expected_value
print(y_base)
shap.summary_plot(shap_values, x_test, max_display=50)
shap.force_plot(explainer.expected_value, shap_values, x_test)
print(cell_line)
