import argparse

import numpy as np
import torch
import torch.nn as nn
import os

parser = argparse.ArgumentParser(description='Extract DNA2vec Features')
parser.add_argument('--input_file',required=True,help='Input file (cell_line.fasta) e.g.data/train/GM12878.fasta')
parser.add_argument('--cell_line',required=True,help='the extracted dataset name e.g. GM12878')
parser.add_argument('--set',required=True,help='the extracted dataset for training or testing e.g. train')
args = parser.parse_args()
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
        feature_seq = []
        for i in range(len(kmer_list[number])):
            kmer_index = kmer_list[number][i]
            for j in word2vec[kmer_index].tolist():
                feature_seq.append(j)
        feature_seq_tensor = torch.Tensor(feature_seq)
        feature_seq_tensor = torch.unsqueeze(feature_seq_tensor, 0)
        feature_seq_tensor = torch.unsqueeze(feature_seq_tensor, 0)
        feature_seq_tensor_avg = nn.AdaptiveAvgPool1d(10000)(feature_seq_tensor)
        feature_seq_numpy = feature_seq_tensor_avg.numpy()
        feature_seq_numpy = np.squeeze(feature_seq_numpy)
        feature_seq_numpy = np.squeeze(feature_seq_numpy)
        feature_seq_numpy = feature_seq_numpy.astype('float64')
        feature_seq_list = feature_seq_numpy.tolist()

        feature_dna2vec.append(feature_seq_list)

    return feature_dna2vec

input_file = args.input_file
cell_line=args.cell_line
set =args.set
f = open('dna2vec/dna2veck3_index.txt', 'r')
index = f.read()
f.close()
index = index.strip().split('\n')
word2vec = np.loadtxt('dna2vec/dna2veck3_vec.txt')
feature_dna2vec = word_embedding(input_file, index, word2vec)
feature_dna2vec = np.array(feature_dna2vec)
print(feature_dna2vec.shape)
print(feature_dna2vec.dtype)
feature_dna2vec = feature_dna2vec.astype('float64')
if not os.path.exists('feature'):
    os.makedirs('feature')
output_file='feature/'+ cell_line+'_'+set+'_dna2vec.txt'
if not os.path.exists(output_file):
    output_file = open(output_file, "w")
    np.savetxt(output_file, feature_dna2vec)
    output_file.close()
