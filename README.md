# MiT4SL

MiT4SL is the first machine learning model for cross cell line prediction of synthetic lethal (SL) gene pairs. It uses a novel architecture of triplet representation learning for (1) gene features based on a knowledge graph and protein sequences, and (2) cell line information by integrating multi-omics data of gene expression, PPI network and protein sequences.

## 1. Overview

![MiT4SL](https://github.com/JieZheng-ShanghaiTech/MiT4SL/blob/main/MiT4SL_overview.png)

## 2. File Tree

```
MiT4SL
│  LICENSE                                  
│  MiT4SL_overview.png
│  README.md
│  
├─data
│  │  
│  ├─BKG
│  │      node_index_dic.json
│  │      
│  ├─Protein_protein
│  │      cell_3_lin_7428protein.csv
│  │      MiT4SL_Cell_3_lines_nx.pkl
│  │      
│  ├─Protein_sequence
│  │      all_node.pkl
│  │      protein_sequence_embedding.pkl
│  │      
│  └─SL_data
│      │  sldata_3lines.csv
│      │  
│      └─Cell_line_adapted
│          │  
│          └─Jurkat_A375
│                      
└─src
    │  Models.py
    │  train_MiT4SL.py
    │  util.py
```
## 3. Main Requirements

To run this project, you will need the following packages:
- Python 3.10.6
- PyTorch  1.12.1 
- Torch-geometric 1.6.0                 
- Torchaudio 0.12.1              
- Torchvision 0.13.1 

## 4. Training and Evaluation

To train our MiT4SL from scratch, navigate to `./src` and run the following command in your terminal:
```shell
python train_MiT4SL.py
```
This will train the MiT4SL model for the cell-line-adapted scenario (Jurkat&A375 -> A549). The well-trained model and its corresponding classification results will be stored in `./result`.

## 5. Notes

Due to the limitation of GitHub's file size (< 100mb), we upload the BKG data (`kgdata.pkl`) into Google Driver (https://drive.google.com/drive/folders/12exswZrjKjgrG7WNO3lgy1K8GKLztG02?usp=sharing). Please place it in the path of `data/BKG` before running.
