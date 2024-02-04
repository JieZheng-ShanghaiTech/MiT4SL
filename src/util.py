import random
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import HGTLoader



 

def Downstream_data_preprocess(args,cv,n_fold,node_type_dict):
    task_data_path=args.Task_data_path
    train_data=pd.read_csv(f"{task_data_path}/{cv}/cv_{n_fold}/train.txt",header=None,sep=' ')
    test_data=pd.read_csv(f"{task_data_path}/{cv}/cv_{n_fold}/test.txt",header=None,sep=' ')
    # test_data=pd.read_csv("/home/siyutao/SL/LukePi_2.0/HGT/synlethkg/tail_data/tail_test_989.csv")
    test_data.columns=[0,1,2]
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
    # low data scenario settings
    if args.do_low_data:
        num_sample=int(train_data.shape[0]*args.train_data_ratio)
        print(num_sample)
        train_data=train_data.sample(num_sample,replace=False,random_state=0)
        train_data.reset_index(inplace=True)
        print(f'train_data.size:{train_data.shape[0]}')

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
    return train_data,test_data,train_mask,test_mask,num_train_node,num_test_node


def Downstream_data_preprocess_cell(args,cv,n_fold,node_type_dict):
    task_data_path=args.Task_data_path
    #/home/siyutao/SL/LukePi_2.0/cell_line/SL/Cell_SL/C1
    
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
    
    # low data scenario settings
    if args.do_low_data:
        num_sample=int(train_data.shape[0]*args.train_data_ratio)
       # print(num_sample)
        train_data=train_data.sample(num_sample,replace=False,random_state=0)
        train_data.reset_index(inplace=True)
        #print(f'train_data.size:{train_data.shape[0]}')

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

def Downstream_data_preprocess_cellseq(args,cv,n_fold,node_type_dict):
    task_data_path=args.Task_data_path
    #/home/siyutao/SL/LukePi_2.0/cell_line/SL/Cell_SL/C1
    train_data=pd.read_csv(f"{task_data_path}/{cv}/sl_train_{n_fold}.csv")
    test_data=pd.read_csv(f"{task_data_path}/{cv}/sl_test_{n_fold}.csv")
    
    # test_data=pd.read_csv("/home/siyutao/SL/LukePi_2.0/HGT/synlethkg/tail_data/tail_test_989.csv")
    if args.task!='SL':
        train_data=train_data[['0','1','2','3']]
        test_data=test_data[['0','1','2','3']]
        if args.task=='mvgcn_SL':
            train_data=train_data[['0','1','2','3']]
            test_data=test_data[['0','1','2','3']]
          

    test_data.columns=[0,1,2,3]
    train_data.columns=[0,1,2,3]
    train_data[0]=train_data[0].astype(int)
    train_data[1]=train_data[1].astype(int)
    test_data[0]=test_data[0].astype(int)
    test_data[1]=test_data[1].astype(int)
   
    # low data scenario settings
    if args.do_low_data:
        num_sample=int(train_data.shape[0]*args.train_data_ratio)
        print(num_sample)
        train_data=train_data.sample(num_sample,replace=False,random_state=0)
        train_data.reset_index(inplace=True)
        print(f'train_data.size:{train_data.shape[0]}')

    train_node=list(set(train_data[0])|set(train_data[1]))
   
   
    test_node=list(set(test_data[0])|set(test_data[1]))
   
  
    
    num_train_node=len(train_node)
    num_test_node=len(test_node)
    
    return train_data,test_data,num_train_node,num_test_node






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


def Construct_loadercell(args,kgdata,train_mask,test_mask,node_type,num_train_node,num_test_node):

    train_loader = HGTLoader(kgdata,
    num_samples={key: [args.sample_nodes] * args.sample_layers for key in kgdata.node_types},shuffle=False,
    batch_size=num_train_node,
    input_nodes=(node_type,train_mask),num_workers=args.num_workers)

    test_loader=HGTLoader(kgdata,
    num_samples={key: [args.sample_nodes] * args.sample_layers for key in kgdata.node_types},
    batch_size=num_test_node,
    input_nodes=(node_type,test_mask),num_workers=args.num_workers,shuffle=False)



    return train_loader,test_loader

def Construct_cellline_loader(args,kgdata,cell_protein_nx,node_type):
    cell_protein=list(set(cell_protein_nx['kg_newid']))
    num_protein_size=len(set(cell_protein_nx['kg_newid']))
    cell_protein_mask=torch.zeros((27671))
    cell_protein_mask[cell_protein]=1
    cell_protein_mask=cell_protein_mask.bool()
    cell_loader = HGTLoader(kgdata,
    num_samples={key: [args.sample_nodes] * args.sample_layers for key in kgdata.node_types},shuffle=False,
    batch_size=num_protein_size,input_nodes=(node_type,cell_protein_mask),
    num_workers=args.num_workers)

    

    return cell_loader












    
