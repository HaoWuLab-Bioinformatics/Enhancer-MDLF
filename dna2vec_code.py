import numpy as np
import torch
import torch.nn as nn
# from keras.layers.convolutional import Conv2D
import os


def word_embedding(filename, index, word2vec):
    f = open(filename, 'r')
    sequence = []
    for line in f.readlines():
        if line[0] != ' ':
            if line[0] != '>':
                sequence.append(line.upper().strip('\n'))

    k = 3
    kmer_list = []

    for number in range(len(sequence)):
        seq = []
        for i in range(len(sequence[number]) - k + 1):
            ind = index.index(sequence[number][i:i + k])
            seq.append(ind)
        kmer_list.append(seq)

    '''sum_length = 0
    cnt = 0
    for number in range(len(sequence)):
        sum_length += (len(sequence[number]) - k + 1)
        cnt = number
    average_length = round(sum_length / (cnt + 1))'''

    feature_dna2vec = []
    for number in range(len(kmer_list)):
        # print(number)
        feature_seq = []
        for i in range(len(kmer_list[number])):
            kmer_index = kmer_list[number][i]
            for j in word2vec[kmer_index].tolist():
                feature_seq.append(j)
        feature_seq_tensor = torch.Tensor(feature_seq)
        feature_seq_tensor = torch.unsqueeze(feature_seq_tensor, 0)
        feature_seq_tensor = torch.unsqueeze(feature_seq_tensor, 0)
        feature_seq_tensor_avg = nn.AdaptiveAvgPool1d(10000)(feature_seq_tensor)
        # feature_seq_numpy = feature_seq_tensor.numpy()

        # print(feature_seq_numpy)
        feature_seq_numpy = feature_seq_tensor_avg.numpy()
        feature_seq_numpy = np.squeeze(feature_seq_numpy)
        feature_seq_numpy = np.squeeze(feature_seq_numpy)
        feature_seq_numpy = feature_seq_numpy.astype('float64')
        feature_seq_list = feature_seq_numpy.tolist()

        feature_dna2vec.append(feature_seq_list)

    return feature_dna2vec


cell_lines = ['NHLF', 'GM12878', 'HEK293', 'HMEC', 'HSMM', 'HUVEC', 'K562', 'NHEK']
f = open('dna2vec/dna2veck3_index.txt', 'r')
index = f.read()
f.close()
index = index.strip().split('\n')
word2vec = np.loadtxt('dna2vec/dna2veck3_vec.txt')
for cell_line in cell_lines:
    for set in ['train', 'test']:
        # for set in ['train']:
        # filename = 'data/' + set + '/' + cell_line + '.fasta'
        filename = 'dataset/' + cell_line + '/enhancers/' + set + '/' + 'enhancers.fasta'
        feature_dna2vec = word_embedding(filename, index, word2vec)
        feature_dna2vec = np.array(feature_dna2vec)
        print(feature_dna2vec.shape)
        print(feature_dna2vec.dtype)
        feature_dna2vec = feature_dna2vec.astype('float64')
        # file = 'EPfeature/dna2veck3/' + cell_line + '_' + set + '_dna2veck3.txt'
        file = 'feature/dna2veck3/' + cell_line + '_' + set + '_dna2veck3.txt'
        np.savetxt(file, feature_dna2vec)
