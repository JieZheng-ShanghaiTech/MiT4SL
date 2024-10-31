# MiT4SL

MiT4SL is the first machine learning model for cross cell line prediction of synthetic lethal (SL) gene pairs. It uses a novel architecture of triplet representation learning for (1) gene features based on a knowledge graph and protein sequences, and (2) cell line information by integrating multi-omics data of gene expression, PPI network and protein sequences.



![Intro](C:/Users/tsy/Desktop/MiT4SL/MiT4SL/fig_intro.png)

## 1.Overview

![MiT4SL](C:/Users/tsy/Desktop/MiT4SL/MiT4SL/MiT4SL_overview.png)

## 2.File Tree

```
MiT4SL
│  LICENSE                
│  MiT4SL_overview.png
│  README.md                     # README file
│  
├─data                      
│  │  
│  ├─kg_data                     # this file provides the BKG data 
│  │      node_index_dic.json    # the node index map dict
│  │  
│  ├─cell_data                   # this file provides the PPI data 
│  │      Multi_6_cell_lines_proteins.csv #specific gene sets 
│  │      Multi_6_cell_lines_subgraph.pkl #sub-PPI graph 
│  │  
│  ├─seq_data                   # this file provides the sequence data 
│  │     protein_sequence_embedding.pkl #the initial embs for genes
│  │     
│  │  
│  └─sl_data                    # this file provides the SL label data 
│      └─Adapted                # cell-line-adapted scenario
│          │  
│          └─Multi_5_to_A549
│    
└─src              
    │  train_MiT4SL.py      #running script for training and testing 
    |  configs.py
    |  models.py 
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
Here, we provide an example (Multi_5_cell_lines -> A549) to demonstrate that how to use MiT4SL. 

To train our MiT4SL from scratch, navigate to `./` and run the following command in your terminal:

```shell
python train_MiT4SL.py --cfg_path ./configs/MiT4SL/configs/MiT4SL_adapted_Multi_5_to_A549.yaml
```

This will train the MiT4SL model for the cell-line-adapted scenario (Multi_5_cell_lines -> A549). The well-trained model and its corresponding classification metrics will be stored in `./result`.

## 5. Notes
Due to the limitation of GitHub's file size (< 100mb), we upload the BKG data (`kgdata.pkl`) into Google Driver (https://drive.google.com/drive/folders/12exswZrjKjgrG7WNO3lgy1K8GKLztG02?usp=sharing). Please place it in the path of `./data/BKG` before running.

