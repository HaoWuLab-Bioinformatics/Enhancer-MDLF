#!/bin/bash
python dna2vec_code.py --input_file data/train/$PARAM1.fasta --cell_line $PARAM1 --set train
python dna2vec_code.py --input_file data/test/$PARAM1.fasta --cell_line $PARAM1 --set test
python motif_find.py --input_file data/train/$PARAM1.fasta --cell_line $PARAM1 --set train
python motif_find.py --input_file data/test/$PARAM1.fasta --cell_line $PARAM1 --set test
python main.py --cell_line $PARAM1