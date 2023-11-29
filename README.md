# Enhancer-MDLF
EnhancerMDLF: A novel deep learning framework for identifying cell-specific enhancers
## Framework
![image](Figure/framework.jpg)
## Overview
The folder "**data**" contains the data of the enhancers, containing the sequences of the independent test sets and training sets on eight cell lines.  
The folder "**EPdata**" contains the data of the enhancer and promoter, containing the sequences of the independent test sets and training sets. The first half of each file is labeled as 0, and the second half is labeled as 1.  
The folder "**generic data**" contains the data of the enhancer created by Liu et al.[1], containing the sequences of the independent test sets and training sets.   
The folder "**model**" contains the trained models on eight cell lines and the pre-trained models are trained on all cell lines for use or validation.  
The folder "**dna2vec**" contains the pre-trained DNA vectors provided in dna2vec[2].  
The folder "**motif**" contains the position weight matrix (PWM) of motifs and the p-value threshold score from the HOCOMOCO Human v11 database[3].  
The file "**network.py**" is the code of the network architecture.  
The file "**main.py**" is the code of the entire model.   
The file "**dna2vec_code.py**" is the code used to extract dna2vec features.  
The file "**motif_find.py**" is the code used to extract motif features.  
The file "**SHAP.py**" is the code for exploring important motifs.  
## Dependency
Python 3.8  
tensorflow 2.2.0  
scikit-learn  
numpy  
See requirements.txt for all detailed libraries  
## Usage
### Step 0. Prepare dataset
We have provided enhancer training and test set data and labels for eight cell lines in the following directory:  
training set data : 'data/train/${cell line name}.fasta'  (**e.g.** 'data/train/GM12878.fasta')  
training set label : 'data/train/${cell line name}_y_train.txt'  (**e.g.** 'data/train/GM12878_y_train.txt')  
test set data : 'data/test/${cell line name}.fasta'  (**e.g.** 'data/test/GM12878.fasta')  
test set label : 'data/test/${cell line name}_y_test.txt'  (**e.g.** 'data/test/GM12878_y_test.txt')  
If users want to run Enhancer-MDLF using their own dataset , please organize the data in the format described above.  
### Step 1. Extract features of enhancers
Before running Enhancer-MDLF,users should extract features of enhancers through run the script to extract dna2vec-based features and motif-based features as follows:  
#### necessary input  
input = 'the data file from which you want to extract features.The file naming format is the same as in step 0.'  
cell_line = 'the cell line name for feature exrtraction'  
set = 'the extracted data for training or testing'  
#### run the script
`python dna2vec_code.py --input_file ${input} --cell_line ${cell_line} --set ${set}`   
**e.g.**`python dna2vec_code.py --input_file data/train/GM12878.fasta --cell_line GM12878 --set train`  
**e.g.**`python dna2vec_code.py --input_file data/test/GM12878.fasta --cell_line GM12878 --set test`  
`python motif_find.py --input_file ${input} --cell_line ${cell_line} --set ${set}`  
**e.g.**`python motif_find.py --input_file data/train/GM12878.fasta --cell_line GM12878 --set train`  
**e.g.**`python motif_find.py --input_file data/test/GM12878.fasta --cell_line GM12878 --set test`  
The output feature files will be saved in the 'feature' directory
### step 2. Run Enhancer-MDLF:  
Users can run the script as follows to compile and run Enhancer-MDLF:    
#### necessary input  
cell_line = 'the cell line name for train and prediction'  
#### run the script
`python main.py --cell_line ${cell_line}`    
e.g.`python main.py --cell_line GM12878`   
## Reference
[1] Liu B, Fang L, Long R, et al. iEnhancer-2L: A two-layer predictor for identifying enhancers and their strength by pseudo k-tuple nucleotide composition. Bioinformatics 2016; 32:362–369  
[2] Ng P. dna2vec: Consistent vector representations of variable-length k-mers. arXiv preprint arXiv 2017;1701.06279
[3] Kulakovskiy I V., Vorontsov IE, Yevshin IS, et al. HOCOMOCO: Towards a complete collection of transcription factor binding models for human and mouse via large-scale ChIP-Seq analysis. Nucleic Acids Res 2018; 46: D252–D259  
