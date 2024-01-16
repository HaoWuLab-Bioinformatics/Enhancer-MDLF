import argparse

import numpy as np
from tqdm import tqdm
import os
parser = argparse.ArgumentParser(description='Extract motif Features')
parser.add_argument('--input_file',required=True,help='Input file (cell_line.fasta) e.g.data/train/GM12878.fasta')
parser.add_argument('--cell_line',required=True,help='the extracted dataset name e.g. GM12878')
parser.add_argument('--set',required=True,help='the extracted dataset for training or testing e.g. train')
args = parser.parse_args()
def motif_compare(seq, motif, threshold):
    num = len(seq)
    seq = seq.upper()
    length = motif.shape[0]
    cnt = 0
    for i in range(num - length + 1):
        score = 0
        seq_list = seq[i:i + length]
        for j in range(len(seq_list)):
            if seq_list[j] == 'A':
                score += motif[j][0]
            elif seq_list[j] == 'C':
                score += motif[j][1]
            elif seq_list[j] == 'G':
                score += motif[j][2]
            elif seq_list[j] == 'T':
                score += motif[j][3]
        if score >= threshold:
            cnt = cnt + 1
    return cnt


fmotif = open("motif/HOCOMOCOv11_core_pwms_HUMAN_mono.txt", 'r')
motifs = {}
for line in fmotif.readlines():
    if line[0] != ' ':
        if line[0] == '>':
            key = line.strip('>').strip('\n')
            a = []
        if line[0] != '>':
            a.append(list(line.upper().strip('\n').split("\t")))
            motifs[key] = a

for key in motifs.keys():
    motifs[key] = np.array(motifs[key], dtype="float64")

fthre = open("motif/HOCOMOCOv11_core_HUMAN_mono_homer_format_0.0001.txt", 'r')
thresholds = {}
key_val = []
for line in fthre.readlines():
    if line[0] != ' ':
        if line[0] == '>':
            key_val = list(line.strip('\n').split("\t"))
            key = key_val[1]
            thresholds[key] = key_val[2]
for key in thresholds.keys():
    thresholds[key] = np.array(thresholds[key], dtype="float64")

input_file = args.input_file
cell_line=args.cell_line
set =args.set
print('Extracting motif features for the '+ set +' set of '+ cell_line)
fseq = open(input_file, 'r')
sequences = []
counts = []
count = 0
for line in fseq.readlines():
    if line[0] != ' ':
        if line[0] != '>':
            sequences.append(line.upper().strip('\n'))
for number in tqdm(range(len(sequences)), desc="Processing data", unit="item"):
#for number in range(len(sequences)):
    for key in motifs.keys():
        count = 0
        sequence = sequences[number]
        motif = motifs[key]
        threshold = thresholds[key]
        count += motif_compare(sequence, motif, threshold)
        counts.append(count)
counts = np.array(counts)
counts = counts.reshape(len(sequences), -1)
if not os.path.exists('feature'):
    os.makedirs('feature')
output_file='feature/'+ cell_line+'_'+set+'_motif.txt'
if not os.path.exists(output_file):
    output_file = open(output_file, "w")
    np.savetxt(output_file, counts)
    output_file.close()
