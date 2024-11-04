import random
import torch
import numpy as np
import pandas as pd
import json
import logging
import os
import pickle
from torch_geometric.loader import HGTLoader
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,average_precision_score


def Downstream_data_preprocess_cell(Task_data_path,Cell,node_type_dict):
    """
    Preprocesses training, validation, and test data for SL prediction task.

    Parameters:
    - args: Argument object containing parameters such as the data path and cell type.
    - node_type_dict: Dictionary mapping node identifiers to node types.

    Returns:
    - train_data, val_data, test_data: Processed training, validation, and test data.
    - train_mask, val_mask, test_mask: Boolean masks indicating nodes present in each dataset.
    - num_train_node, num_val_node, num_test_node: Counts of unique nodes in each dataset.
    - ori_train_data, ori_val_data, ori_test_data: Original copies of the processed datasets.
    """

    # Load train and test data from CSV files
    train_data=pd.read_csv(f"{Task_data_path}/{Cell}/sl_train_0.csv")
    test_data=pd.read_csv(f'{Task_data_path}/{Cell}/sl_test_0.csv')
    # Split training data into train and validation sets (90% train, 10% validate)
    all_idx=list(range(len(train_data)))
    random.seed(0)  
    random.shuffle(all_idx)         
    split = int(0.9 * len(all_idx))
    train_idx, valid_idx = all_idx[:split], all_idx[split:]
    # Separate data for training and validation
    val_data=train_data.loc[valid_idx]
    val_data.reset_index(drop=True,inplace=True)
    train_data=train_data.loc[train_idx]
    train_data.reset_index(drop=True,inplace=True)
    ori_train_data, ori_test_data, ori_val_data = train_data.copy(), test_data.copy(), val_data.copy()
    # Standardize column names across datasets for easier access
    for df in [ori_train_data, ori_val_data, ori_test_data, train_data, val_data, test_data]:
        df.columns = [0, 1, 2, 3]
    ori_test_data=ori_test_data.drop_duplicates()
    ori_test_data.reset_index(drop=True,inplace=True)
    # Map node IDs (PrimeKG_index) to their corresponding types using node_type_dict
    for df in [train_data, val_data, test_data]:
        df[0] = df[0].astype(str).map(node_type_dict)
        df[1] = df[1].astype(str).map(node_type_dict)
    train_data=train_data.dropna()
    val_data=val_data.dropna()
    test_data=test_data.dropna()
    # Convert node columns back to integer type
    for df in [train_data, val_data, test_data]:
        df[0] = df[0].astype(int)
        df[1] = df[1].astype(int)
    
    test_data=test_data.drop_duplicates()
    test_data.reset_index(drop=True,inplace=True)
     # Generate unique nodes for train, validation, and test sets
    train_node = list(set(train_data[0]) | set(train_data[1]))
    val_node = list(set(val_data[0]) | set(val_data[1]))
    test_node = list(set(test_data[0]) | set(test_data[1]))

    # Initialize boolean masks for each node set
    # the total number of gene in the BKG is 27671
    node_count = 27671
    train_mask, val_mask, test_mask = torch.zeros((node_count), dtype=torch.bool), torch.zeros((node_count), dtype=torch.bool), torch.zeros((node_count), dtype=torch.bool)
    # Set mask values for nodes present in each dataset
    train_mask[train_node] = True
    val_mask[val_node] = True
    test_mask[test_node] = True
    # Count unique nodes in each dataset
    num_train_node = len(train_node)
    num_val_node = len(val_node)
    num_test_node = len(test_node)
    return train_data,val_data,test_data,train_mask,val_mask,test_mask,num_train_node,num_val_node,num_test_node,ori_train_data,ori_val_data,ori_test_data



def Construct_loader(SAMPLE_NODES,SAMPLE_LAYERS,NUM_WORKERS,kgdata,train_mask,val_mask,test_mask,node_type,num_train_node,num_val_node,num_test_node):
    """
        Constructs data loaders for training, validation, and testing with heterogeneous graph sampling.

        Parameters:
        - args: Argument object containing parameters like sample nodes, sample layers, and num_workers.
        - kgdata: Knowledge graph data to be loaded.
        - train_mask, val_mask, test_mask: Boolean masks indicating nodes present in each dataset.
        - node_type: Type of node to be used as input for each loader.
        - num_train_node, num_val_node, num_test_node: Number of nodes for batch size in each dataset.

        Returns:
        - train_loader, val_loader, test_loader: Data loaders for training, validation, and testing.
    """
    # Construct the training loader
    train_loader = HGTLoader(kgdata,
    num_samples={key: [SAMPLE_NODES] * SAMPLE_LAYERS for key in kgdata.node_types},shuffle=False,
    batch_size=num_train_node,
    input_nodes=(node_type,train_mask),num_workers=NUM_WORKERS)

    # Construct the validation loader
    val_loader = HGTLoader(kgdata,
    num_samples={key: [SAMPLE_NODES] * SAMPLE_LAYERS for key in kgdata.node_types},shuffle=False,
    batch_size=num_val_node,
    input_nodes=(node_type,val_mask),num_workers=NUM_WORKERS)

    # Construct the test loader
    test_loader=HGTLoader(kgdata,
    num_samples={key: [SAMPLE_NODES] * SAMPLE_LAYERS for key in kgdata.node_types},
    batch_size=num_test_node,
    input_nodes=(node_type,test_mask),num_workers=NUM_WORKERS,shuffle=False)

    return train_loader,val_loader,test_loader



def set_seed(seed):  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model(model, optimizer_model,SAVE_MODEL_PATH):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    torch.save({
        
        'model_state_dict': model.state_dict(),
        'optimizer_model_state_dict': optimizer_model.state_dict()},
        os.path.join(SAVE_MODEL_PATH, 'checkpoint')
    )
                 
def set_logger(args):
    '''
    Write logs to checkpoint and console 
    '''
    log_file = os.path.join(args.Save_model_path, 'train.log') 
    
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s', 
        level=logging.INFO,  # 
        datefmt='%Y-%m-%d %H:%M:%S', 
        filename=log_file, 
        filemode='w'  
    )
    console = logging.StreamHandler() # 
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s') 
    console.setFormatter(formatter) 
    logging.getLogger('').addHandler(console) 




def compute_accuracy(target,pred, pred_edge):
    
    target=target.clone().detach().cpu().numpy()
    pred=pred.clone().detach().cpu().numpy()
    pred_edge=pred_edge.clone().detach().cpu()
    scores = torch.softmax(pred_edge, 1).numpy()
    target=target.astype(int)
   
    aucu=roc_auc_score(target,scores[:,1])
    aupr=average_precision_score(target, scores[:,1])
    bacc=balanced_accuracy_score(target,pred)
    return aucu,aupr,bacc



    
def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode,metric,step, metrics[metric]))






def init_graph_data(KG_DATAPATH,CELLNX_DATAPATH):
    with open (KG_DATAPATH,'rb') as f:
        kgdata=pickle.load(f)
    
    num_nodes_type=len(kgdata.node_types)

    input_node_embeddings = torch.nn.Embedding(num_nodes_type, 16)
    torch.nn.init.xavier_uniform_(input_node_embeddings.weight.data)
    for i in range(len(kgdata.node_types)):
        num_repeat=kgdata[kgdata.node_types[i]].x.shape[0]
        kgdata[kgdata.node_types[i]].x =input_node_embeddings(torch.tensor(i)).repeat([num_repeat,1]).detach()
    
    with open(CELLNX_DATAPATH,'rb') as f:
        cell_ppidata_list=pickle.load(f)
    cell_ppidata=[]
    for i in cell_ppidata_list:
        input_node_embeddings = torch.nn.Embedding(i.x.shape[0], 16)
        torch.nn.init.xavier_uniform_(input_node_embeddings.weight.data)
        i.x=input_node_embeddings.weight.data
        cell_ppidata.append(i)
    return kgdata,cell_ppidata




def overlapping_with_sequence(PROTEINSeq_DATAPATH,CELLPROTEIN_DATAPATH,Task_data_path,cell):
    with open(PROTEINSeq_DATAPATH,'rb') as f:
        proteinseq_data=pickle.load(f)

    cell_line_proteins=pd.read_csv(CELLPROTEIN_DATAPATH)
    train_data=pd.read_csv(f'{Task_data_path}/{cell}/sl_train_0.csv')
    test_data=pd.read_csv(f'{Task_data_path}/{cell}/sl_test_0.csv')
    train_node=set(train_data['0'])|set(train_data['1'])
    test_node=set(test_data['0'])|set(test_data['1'])
    slnode=train_node|test_node
    k1=set(cell_line_proteins['primekg_index'])
    all_node=slnode|k1
    new_proteinseq_data={}
    for i in all_node:
        new_proteinseq_data[i]=proteinseq_data[i] 
    
    del proteinseq_data,train_data,test_data
    return new_proteinseq_data,cell_line_proteins

