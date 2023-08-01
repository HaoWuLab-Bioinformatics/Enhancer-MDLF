from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, AveragePooling1D
from tensorflow.keras.models import load_model
import os
import random
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, concatenate, Conv1D, MaxPooling1D, Flatten, Dropout, Layer, Reshape, \
    Concatenate, \
    LeakyReLU, GlobalAveragePooling1D, GlobalMaxPooling1D, ReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold, GroupKFold
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import LSTM, BatchNormalization


class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def set_seed(seed):
    tf.random.set_seed(seed)
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def CNN(data_shape, kernel_num, kernel_size, pool_size, dropout_rate):
    input_data = Input(shape=data_shape)
    x = Conv1D(16, kernel_size=9, strides=2, activation='relu', padding='same')(input_data)
    x = MaxPooling1D(pool_size=pool_size)(x)

    x = Conv1D(16, kernel_size=9, strides=2, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Conv1D(16, kernel_size=9, strides=2, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)

    # x = Conv1D(kernel_num, kernel_size=kernel_size, strides=1, activation='relu', padding='same')(x)
    # x = MaxPooling1D(pool_size=pool_size)(x)

    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input_data, output)
    print(model.summary())
    return model


def cnn_dna2vec(data_shape_dna2vec, kernel_num, kernel_size, pool_size, dropout_rate1, stride):
    input_data_dna2vec = Input(shape=data_shape_dna2vec)
    x_dna2vec = Conv1D(kernel_num, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(
        input_data_dna2vec)
    x_dna2vec = MaxPooling1D(pool_size=pool_size)(x_dna2vec)
    x_dna2vec = Conv1D(kernel_num, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(
        x_dna2vec)
    x_dna2vec = MaxPooling1D(pool_size=pool_size)(x_dna2vec)
    x_dna2vec = Conv1D(kernel_num, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(
        x_dna2vec)
    x_dna2vec = MaxPooling1D(pool_size=pool_size)(x_dna2vec)
    x_dna2vec = Dropout(dropout_rate1)(x_dna2vec)
    x_dna2vec = Flatten()(x_dna2vec)
    x_dna2vec = Dense(500, activation='relu')(x_dna2vec)
    output = Dense(1, activation='sigmoid')(x_dna2vec)
    model = Model(input_data_dna2vec, output)
    print(model.summary())
    return model


def linear_motif(data_shape_motif, dropout_rate1):
    input_data_motif = Input(shape=data_shape_motif)
    x_motif = Dense(128, activation='relu')(input_data_motif)
    x_motif = Dense(64, activation='relu')(x_motif)
    x_motif = Dense(16, activation='relu')(x_motif)
    x_motif = Dropout(dropout_rate1)(x_motif)
    output = Dense(1, activation='sigmoid')(x_motif)
    model = Model(input_data_motif, output)
    print(model.summary())
    return model



def Enhancer_MDLF(data_shape_dna2vec, data_shape_motif, kernel_num, kernel_size, pool_size, dropout_rate1,
                  dropout_rate2, stride):
    input_data_dna2vec = Input(shape=data_shape_dna2vec)
    x_dna2vec = Conv1D(kernel_num, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(
        input_data_dna2vec)
    x_dna2vec = MaxPooling1D(pool_size=pool_size)(x_dna2vec)
    x_dna2vec = Conv1D(kernel_num, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(
        x_dna2vec)
    x_dna2vec = MaxPooling1D(pool_size=pool_size)(x_dna2vec)
    x_dna2vec = Conv1D(kernel_num, kernel_size=kernel_size, strides=stride, activation='relu', padding='same')(
        x_dna2vec)
    x_dna2vec = MaxPooling1D(pool_size=pool_size)(x_dna2vec)
    x_dna2vec = Dropout(dropout_rate1)(x_dna2vec)
    x_dna2vec = Flatten()(x_dna2vec)
    x_dna2vec = Dense(500, activation='relu')(x_dna2vec)

    input_data_motif = Input(shape=data_shape_motif)
    x_motif = Dense(128, activation='relu')(input_data_motif)
    x_motif = Dense(64, activation='relu')(x_motif)
    x_motif = Dense(16, activation='relu')(x_motif)
    x_motif = Dropout(dropout_rate1)(x_motif)

    merge1 = Concatenate(axis=1)([x_dna2vec, x_motif])
    merge1 = Dropout(dropout_rate2)(merge1)
    output = Dense(1, activation='sigmoid')(merge1)
    model = Model(
        [input_data_dna2vec, input_data_motif],
        output)
    print(model.summary())
    return model



