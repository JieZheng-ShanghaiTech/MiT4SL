import random
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import HGTLoader


def Downstream_data_preprocess_cell(args,cv,n_fold,node_type_dict):
    """
    To load the SL data to HGT_loader, we map the gene index (primekg_index) to the new heterogenous graph index by node_type_dict
    Args:
        cv:(str) evaluation scenario. (defalut: C2 (cell line adapted))
        n_fold: the number of fold.
        node_type_dict: a dict which can map the primekg_index to the new heterogeous graph index.
    """
    task_data_path=args.Task_data_path
    if args.train_data_ratio!=0:
        train_data=pd.read_csv(f"{task_data_path}/sl_train_{n_fold}.csv")
        test_data=pd.read_csv(f"{task_data_path}/sl_test_{n_fold}.csv")
    else:
        if args.cv=='C2':

            train_data=pd.read_csv(f"{task_data_path}/{cv}/sl_train_0.csv")
            test_data=pd.read_csv(f"{task_data_path}/{cv}/sl_test_0.csv")
        else:
            train_data=pd.read_csv(f"{task_data_path}/sl_train_{n_fold}.csv")
            test_data=pd.read_csv(f"{task_data_path}/sl_test_{n_fold}.csv")
          
    ori_train_data=train_data.copy()
    ori_test_data=test_data.copy()
    ori_train_data.columns=[0,1,2,3]
    ori_test_data.columns=[0,1,2,3]
    if args.do_low_data:
        num_sample=int(ori_train_data.shape[0]*args.train_data_ratio)
        ori_train_data=ori_train_data.sample(num_sample,replace=False,random_state=0)
        ori_train_data.reset_index(inplace=True)
    test_data.columns=[0,1,2,3]
    train_data.columns=[0,1,2,3]
    
    train_data[0]=train_data[0].astype(str).map(node_type_dict)
    train_data[1]=train_data[1].astype(str).map(node_type_dict)
    test_data[0]=test_data[0].astype(str).map(node_type_dict)
    test_data[1]=test_data[1].astype(str).map(node_type_dict)
    
    train_data=train_data.dropna()
    test_data=test_data.dropna()
   
    train_data[0]=train_data[0].astype(int)
    train_data[1]=train_data[1].astype(int)
    test_data[0]=test_data[0].astype(int)
    test_data[1]=test_data[1].astype(int)
    
    train_node=list(set(train_data[0])|set(train_data[1]))
    train_mask=torch.zeros((27671))
    test_mask=torch.zeros((27671))
    test_node=list(set(test_data[0])|set(test_data[1]))
   
    
    train_mask[train_node]=1
    test_mask[test_node]=1
   
    train_mask=train_mask.bool()
    test_mask=test_mask.bool()
    
    num_train_node=len(train_node)
    num_test_node=len(test_node)
    
    return train_data,test_data,train_mask,test_mask,num_train_node,num_test_node,ori_train_data,ori_test_data




def Construct_loader(args,kgdata,train_mask,test_mask,node_type,num_train_node,num_test_node):
    
    train_loader = HGTLoader(kgdata,
    num_samples={key: [args.sample_nodes] * args.sample_layers for key in kgdata.node_types},shuffle=False,
    batch_size=num_train_node,
    input_nodes=(node_type,train_mask),num_workers=args.num_workers)

    test_loader=HGTLoader(kgdata,
    num_samples={key: [args.sample_nodes] * args.sample_layers for key in kgdata.node_types},
    batch_size=num_test_node,
    input_nodes=(node_type,test_mask),num_workers=args.num_workers,shuffle=False)

    

    return train_loader,test_loader














    
