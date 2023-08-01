import numpy as np


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
print(motifs)

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
print(thresholds)

# cell_lines = ['NHLF', 'GM12878', 'HEK293', 'HMEC', 'HSMM', 'HUVEC', 'K562', 'NHEK', 'enhancerdata']
# cell_lines = ['enhancerdata']
cell_lines = ['HMEC']
for cell_line in cell_lines:
    for set in ['train', 'test']:
        # for set in ['test']:
        filename = 'data/' + set + '/' + cell_line + '.fasta'
        fseq = open(filename, 'r')
        sequences = []
        counts = []
        count = 0
        for line in fseq.readlines():
            if line[0] != ' ':
                if line[0] != '>':
                    sequences.append(line.upper().strip('\n'))
        for number in range(len(sequences)):
            for key in motifs.keys():
                count = 0
                sequence = sequences[number]
                motif = motifs[key]
                threshold = thresholds[key]
                count += motif_compare(sequence, motif, threshold)

                counts.append(count)
            print(count)
        counts = np.array(counts)
        counts = counts.reshape(len(sequences), -1)
        print(counts)
        print(counts.shape)
        file = 'feature/motif/' + cell_line + '_' + set + '_motif.txt'
        np.savetxt(file, counts)
