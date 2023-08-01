import os

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
import network
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, \
    recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import matthews_corrcoef
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import os

def EvaluateMetrics(y_test, proba, label):
    acc = accuracy_score(y_test, label)
    fpr, tpr, thresholdTest = roc_curve(y_test, proba)
    aucv = auc(fpr, tpr)
    bacc = balanced_accuracy_score(y_test, label)
    sn = recall_score(y_test, label)
    tn, fp, fn, tp = confusion_matrix(y_test, label).ravel()
    sp = tn / (tn + fp)
    mcc = matthews_corrcoef(y_test, label)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, label, average='binary')
    print(aucv, '\t', bacc, '\t', mcc, '\t', sn, '\t', sp, '\t', fscore)


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


network.set_seed(2023)
cell_lines = 'GM12878'
file_train = 'data/train/' + cell_lines + '.fasta'
x_train_dna2vec = np.loadtxt('feature/dna2veck3/' + cell_lines + '_train_dna2veck3.txt', )
x_train_motif = np.loadtxt('feature/motifcount/pvalue0.0001/' + cell_lines + '_train_motif.txt')
x_train_motif = noramlization(x_train_motif, file_train)
y_train = np.loadtxt('data/train/' + cell_lines + '_y_train.txt')

file_test = 'data/test/' + cell_lines + '.fasta'
x_test_dna2vec = np.loadtxt('feature/dna2veck3/' + cell_lines + '_test_dna2veck3.txt')
x_test_motif = np.loadtxt('feature/motifcount/pvalue0.0001/' + cell_lines + '_test_motif.txt')
x_test_motif = noramlization(x_test_motif, file_test)
y_test = np.loadtxt('data/test/' + cell_lines + '_y_test.txt')

x_train_dna2vec = np.expand_dims(x_train_dna2vec, 2)
x_test_dna2vec = np.expand_dims(x_test_dna2vec, 2)

print(x_train_dna2vec.shape)
print(x_train_motif.shape)
print(y_train.shape)
print(x_test_dna2vec.shape)
print(x_test_motif.shape)
print(y_test.shape)
data_shape_dna2vec = x_train_dna2vec.shape[1:3]
data_shape_motif = x_train_motif.shape[1:2]

# parameter setting
learning_rate = 0.0001
kernel_num = 64
kernel_size = 7
pool_size = 2
alpha = 0.5
gamma = 3
MAX_EPOCH = 200
BATCH_SIZE = 100
dropout_rate1 = 0.6
dropout_rate2 = 0.5
stride = 3

LOSS = binary_focal_loss(alpha, gamma)
model = network.Enhancer_MDLF(data_shape_dna2vec, data_shape_motif, kernel_num, kernel_size, pool_size,
                              dropout_rate1,
                              dropout_rate2, stride)
filepath = "model/" + cell_lines + "_model_cnn.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10)
model.compile(loss=LOSS, optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
kf = StratifiedKFold(5, shuffle=True, random_state=10).split(x_train_motif, y_train)
for j, (train_index, test_index) in enumerate(kf):
    history = model.fit([x_train_dna2vec[train_index], x_train_motif[train_index]],
                        y=y_train[train_index],
                        batch_size=BATCH_SIZE, epochs=MAX_EPOCH,
                        validation_data=(
                            [x_train_dna2vec[test_index], x_train_motif[test_index]],
                            y_train[test_index]),
                        callbacks=[checkpoint, early_stopping_monitor])
proba = model.predict(x=[x_test_dna2vec, x_test_motif])
label = np.around(proba)
EvaluateMetrics(y_test, proba, label)
