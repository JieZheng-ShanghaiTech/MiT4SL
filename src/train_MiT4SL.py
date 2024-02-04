import argparse
import json
import logging
from time import time
import os
import torch_geometric.transforms as T
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from Models import *
import pandas as pd
from util import Downstream_data_preprocess_cell,Construct_loader
import pickle
import math
import torch.nn.init as init
from sklearn.metrics import roc_auc_score,f1_score, roc_auc_score,precision_recall_curve,auc,balanced_accuracy_score,precision_score

import pickle
from info_nce import InfoNCE, info_nce


def save_model(model, optimizer_model,save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    print(args.Save_model_path)
    with open(os.path.join(args.Save_model_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_model_state_dict': optimizer_model.state_dict()},
        os.path.join(args.Save_model_path, 'checkpoint')
    )
                 
def set_logger(args):
    '''
    Write logs to checkpoint and console 
    '''

    if args.do_train:
        # train_log=str(linear_layer_count)+'_'+args.lr+'_'+'train.log'
        log_file = os.path.join(args.Save_model_path or args.init_checkpoint, 'train.log') 
    else:
        log_file = os.path.join(args.Save_model_path or args.init_checkpoint, 'test.log') 
    
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
    '''
    Compute the evaluation metrics
    '''
    target=target.clone().detach().cpu().numpy()
    pred=pred.clone().detach().cpu().numpy()
    pred_edge=pred_edge.clone().detach().cpu()
    scores = torch.softmax(pred_edge, 1).numpy()
    target=target.astype(int)
   
    aucu=roc_auc_score(target,scores[:,1])
    precision_tmp, recall_tmp, _thresholds = precision_recall_curve(target, pred)
    aupr = auc(recall_tmp, precision_tmp)
    f1 = f1_score(target,pred)
    bacc=balanced_accuracy_score(target,pred)
  
    
    
    return aucu,aupr,f1,bacc


def train(args,Context_MiT4SL, train_loader, optimizer_model,sldata,num_train_node,ori_train_data,device):
    '''
    Train MiT4SL
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer_model=optimizer_model
    Context_MiT4SL.train()

    batch_size=num_train_node
    loss_sum = 0
    aucu_sum=0
    aupr_sum=0
    f1_sum=0
    bacc_sum=0
   
   
    loss_cl=0
    loss_mse=0
    infoloss=InfoNCE()
    mseloss = nn.MSELoss()
    
    # step_size:30 0.5 for cell line adapted scenario;
    # step_size:80 0.5 for cell line specific scenario;
    scheduler_model=lr_scheduler.StepLR(optimizer_model,step_size=30,gamma=0.5)
    
    for step,batch in enumerate(tqdm(train_loader, desc="Iteration")):
       
        batch = batch.to(device)
        if args.task=='GI_score':
           batch_sl=sldata[sldata[3]==1]
           batch_nosl=sldata[sldata[3]==0]
           batch_nosl.reset_index(drop=True,inplace=True)
           nosl_idx=list(range(batch_nosl.shape[0]))
           np.random.shuffle(nosl_idx)
           sampled_idx=np.random.choice(nosl_idx,size=batch_sl.shape[0]*3,replace=False)
           batch_nosl=batch_nosl.iloc[sampled_idx,:]
           sldata=pd.concat([batch_sl,batch_nosl],axis=0)
           sldata.reset_index(drop=True,inplace=True)
           ori_batch_sl=ori_train_data[ori_train_data[3]==1]
           ori_batch_nosl=ori_train_data[ori_train_data[3]==0]
           ori_batch_nosl=ori_batch_nosl.iloc[sampled_idx,:]
           ori_train_data=pd.concat([ori_batch_sl,ori_batch_nosl],axis=0)
           ori_train_data.reset_index(drop=True,inplace=True)
        prediction_label=sldata[3]
        all_prediction_label=prediction_label
        all_prediction_label=torch.tensor(all_prediction_label.values).to(device)
      
        if args.act=='KGSeq_finalCL':
            tri_emb1,tri_emb2,prediction_result,avergae_prediction=Context_MiT4SL(sldata,args.act,batch,batch_size,ori_train_data)
            loss_cl=infoloss(tri_emb1,tri_emb2)
            predicted_labels = torch.argmax(avergae_prediction, dim=1)
            loss_mse=mseloss(predicted_labels.float(),all_prediction_label.float())

        else:
            prediction_result=Context_MiT4SL(sldata,args.act,batch,batch_size,ori_train_data)
        
        all_prediction_result=prediction_result
        optimizer_model.zero_grad()
        
        loss = criterion(all_prediction_result,all_prediction_label.long())
        loss+=args.beta1*loss_cl+args.beta2*loss_mse

        all_prediction=torch.max(all_prediction_result.detach(),dim=1)[1]
        aucu,aupr,f1,bacc=compute_accuracy(all_prediction_label,all_prediction,all_prediction_result)
        loss.backward()
        optimizer_model.step()
        scheduler_model.step()
       
        loss_sum += float(loss.cpu().item())
        aucu_sum+=float(aucu)
        aupr_sum+=float(aupr)
        f1_sum+=float(f1)
        bacc_sum+=float(bacc)
       
       
        log = {
            'loss': loss_sum/(step+1),
            # 'auc':aucu_sum/(step+1),
            # 'aupr':aupr_sum/(step+1),
            # 'f1':f1_sum/(step+1),
            # 'bacc':bacc_sum/(step+1),
      
           
        }
   
    return log




def eval(args,Context_MiT4SL,test_loader,sldata,num_test_node,ori_test_data,device):
    '''
    Test MiT4SL
    '''
    Context_MiT4SL.eval()
    aucu_sum=0
    aupr_sum=0
    f1_sum=0
    bacc_sum=0
    

    batch_size=num_test_node
    with torch.no_grad():
        for step,batch in enumerate(test_loader):
            batch = batch.to(device)
            prediction_label=sldata[3]
    
            if args.act=='KGSeq_finalCL':
                _,_,prediction_result,_=Context_MiT4SL(sldata,args.act,batch,batch_size,ori_test_data)
            else:
                prediction_result=Context_MiT4SL(sldata,args.act,batch,batch_size,ori_test_data)

            all_prediction_label=prediction_label
            all_prediction_result=prediction_result
            all_prediction=torch.max(all_prediction_result.detach(),dim=1)[1]
            all_prediction_label=torch.tensor(all_prediction_label.values).to(device)
            aucu,aupr,f1,bacc=compute_accuracy(all_prediction_label,all_prediction,all_prediction_result)
            aucu_sum+=float(aucu)
            aupr_sum+=float(aupr)
            f1_sum+=float(f1)
            bacc_sum+=float(bacc)
            
            log = {
            'auc':aucu_sum/(step+1),
            'aupr':aupr_sum/(step+1),
            'f1':f1_sum/(step+1),
            'bacc':bacc_sum/(step+1),
           
          
        }
        return aucu_sum/(step+1),aupr_sum/(step+1),f1_sum/(step+1),bacc_sum/(step+1),log
        


def override_config(args):
    '''
    Override model and data configuration 
    '''
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.method=argparse_dict['method']
    # args.epochs = argparse_dict['epochs']
    args.lr = argparse_dict['lr']
    args.num_layer = argparse_dict['num_layer']
    args.emb_dim = argparse_dict['emb_dim']
    args.act=argparse_dict['act']
    args.epoch=argparse_dict['epoch']
    
    args.gnn_type=argparse_dict['gnn_type']

    if args.Save_model_path is None:
        args.Save_model_path = argparse_dict['Save_model_path']





def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at epoch %d: %f' % (mode,metric,step, metrics[metric]))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--do_train', default=1,type=int)
    parser.add_argument('--do_low_data',default=0,type=int)
    parser.add_argument('--train_data_ratio', default=0,type=float)
    parser.add_argument('--Use_kg', default=1,type=int)
    parser.add_argument('--Use_seq', default=1,type=int)
    parser.add_argument('--Use_cellnx', default=1,type=int)
    parser.add_argument('--Use_omics', default=0,type=int)
    parser.add_argument('--tpm_value', default=400,type=int)
    parser.add_argument('--act', default='KGSeq_finalCL',type=str,help='KGSeq_finalCL')
    parser.add_argument('--cv', default='C2',type=str)
    parser.add_argument('--task', default='GI_score',type=str)
    parser.add_argument('--cell', default='Jurkat_A375',type=str)
    parser.add_argument('--node_type', default='gene/protein',type=str)
    parser.add_argument('--Task_data_path',default='./data/SL_data/Cell_line_adapted/Jurkat_A375',type=str,help='Data filename to input')
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--KG_data_path',default='./data/BKG/kgdata.pkl', type=str)
    parser.add_argument('--Cell_nx_data_path',default='./data/Protein_protein/MiT4SL_Cell_3_lines_nx.pkl', type=str)
    parser.add_argument('--Seq_data_path',default='./data/Protein_sequence/protein_sequence_embedding.pkl', type=str)
    parser.add_argument('--Node_index_path',default=None, type=str)
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of epochs to train ')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.2,
                        help='beta2 (default: 0.2)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 3).')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='embedding dimensions (default: 64)')
    parser.add_argument('--KG_hidden_channels', type=int, default=64,
                        help='number of KG encoder hidden channels.')
    parser.add_argument('--KG_num_heads', type=int, default=4,
                        help='number of KG encoder head.')
    parser.add_argument('--KG_num_layers', type=int, default=3,
                        help='number of KG encoder layer.')
    parser.add_argument('--CellLineGraph_hidden_channels', type=int, default=64,
                        help='number of Cell encoder hidden channels.')
    parser.add_argument('--CellLineGraph_num_layers', type=int, default=15,
                        help='number of Cell encoder layer.')
 
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--sample_nodes', type=int, default=512,
                        help='the number of sampled nodes for each type ')
    parser.add_argument('--sample_layers', type=int, default=4,
                        help='the number of sampled iterations ')
    parser.add_argument('--save_checkpoint_steps', default=10, type=int)
    parser.add_argument('--cell_line_ppi_path',default='data/Protein_protein/cell_3_lin_7428protein.csv',type=str)
    parser.add_argument('--KG',default='PrimeKG',type=str)
    parser.add_argument('--Seq',default='ESM2',type=str)
    parser.add_argument('--Data_used',default='KG_Seq',type=str)
    parser.add_argument('--log_steps', default=1, type=int, help='train log every xx steps')
    parser.add_argument('--Save_model_path', default='./result',type=str, help='filename to output the model')
    parser.add_argument('--num_workers', type=int, default =16, help='number of workers for dataset loading')
    args = parser.parse_args()
    
   
    
    args.Save_model_path=args.Save_model_path+'/'+args.task+'_'+args.act+'/'+args.KG+'_'+args.Seq+'_'+str(args.lr)+'_'+str(args.epochs)+'/'+args.cell+'_'+args.cv+'_'+str(args.beta1)+'_'+str(args.beta2)+'_'+str(args.tpm_value)
    
         
    
    if (not args.do_train): 
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:  
        override_config(args)
        
    elif args.Task_data_path is None: 
        raise ValueError('one of init_checkpoint/data_path must be choosed.')
    if args.do_train and args.Save_model_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.Save_model_path and not os.path.exists(args.Save_model_path): 
            os.makedirs(args.Save_model_path)
    
    set_logger(args)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # device=torch.device("cpu")
  
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    
    

    #load datasets  
    
    with open (args.KG_data_path,'rb') as f:

        kgdata=pickle.load(f)
    
    
    with open("./data/BKG/node_index_dic.json",'rb') as f:
        node_index=json.load(f)

    gene_protein=node_index[args.node_type]
 
    eval_metric_folds={'fold':[],'auc':[],'aupr':[],'f1':[],'bacc':[]}
    node_type=args.node_type
    num_nodes_type=len(kgdata.node_types)
    num_edge_type=len(kgdata.edge_types)
    num_nodes=kgdata.num_nodes
    input_node_embeddings = torch.nn.Embedding(num_nodes_type, 16)
    torch.nn.init.xavier_uniform_(input_node_embeddings.weight.data)
    for i in range(len(kgdata.node_types)):
        num_repeat=kgdata[kgdata.node_types[i]].x.shape[0]
        kgdata[kgdata.node_types[i]].x =input_node_embeddings(torch.tensor(i)).repeat([num_repeat,1]).detach()
    

    with open(args.Cell_nx_data_path,'rb') as f:
        cell_ppidata_list=pickle.load(f)
    cell_ppidata=[]
    for i in cell_ppidata_list:
        input_node_embeddings = torch.nn.Embedding(i.x.shape[0], 16)
        torch.nn.init.xavier_uniform_(input_node_embeddings.weight.data)
        i.x=input_node_embeddings.weight.data
        cell_ppidata.append(i)

    with open(args.Seq_data_path,'rb') as f:
        proteinseq_data=pickle.load(f)
    
    # cell_line_proteins=pd.read_csv("/home/siyutao/SL/LukePi_2.0/cell_line/KG_Sequence/cell_3_lin_7428protein.csv")
    cell_line_proteins=pd.read_csv(args.cell_line_ppi_path)

    
    with open("./data/Protein_sequence/all_node.pkl",'rb') as f:
        all_node=pickle.load(f)
    new_proteinseq_data={}
    for i in all_node:
        new_proteinseq_data[i]=proteinseq_data[i] 
    
    del proteinseq_data


    


    logging.info('KG data: %s' % args.KG) 
    logging.info('Seq data: %s' % args.Seq) 
    logging.info('Cell_line_proteins: %s' %args.cell) 
    logging.info('Data_used: %s' %args.Data_used)
    logging.info('Action: %s' % args.act)
    # initiliaze the model
    for i in range(5):
        logging.info(f'Fold_{i} training...')
        n_fold=i
        #set up models, one for pre-training and one for context embeddings
        
        Context_MiT4SL=ContextMiT4SL(kgdata,cell_ppidata,new_proteinseq_data,cell_line_proteins,
                                     args.KG_hidden_channels, args.emb_dim, 
                                     args.KG_num_heads, args.KG_num_layers,args.CellLineGraph_hidden_channels, args.CellLineGraph_num_layers,
                                     args.Use_kg,args.Use_cellnx,args.Use_seq,device)
       
    
        #set up optimizers
        optimizer_model = optim.Adam(Context_MiT4SL.parameters(), lr=args.lr, weight_decay=args.decay)
 
    
        if args.init_checkpoint: 
            # Restore model from checkpoint directory  
            logging.info('Loading checkpoint %s...' % args.init_checkpoint)
            checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
            init_step = checkpoint['step']
            Context_MiT4SL.load_state_dict(checkpoint['model_state_dict'])
            if args.do_train:
                optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
                # optimizer_linear_pred_edges.load_state_dict(checkpoint['optimizer_classification_state_dict'])

        else:
            logging.info('Ramdomly Initializing Context_MiT4SL Model...')  
            init_step = 0

        step = init_step 
        
        train_data,test_data,train_mask,test_mask,num_train_node,num_test_node,ori_train_data,ori_test_data=Downstream_data_preprocess_cell(args,args.cv,n_fold,gene_protein)

        train_loader,test_loader=Construct_loader(args,kgdata,train_mask,test_mask,node_type,num_train_node,num_test_node)
      
        logging.info('Start Training...')
        logging.info('init_step = %d' % init_step)
        logging.info('num_train_node = %d' % num_train_node)
        logging.info('num_test_node = %d' % num_test_node)
        if args.do_train:
            logging.info('learning_rate = %f' %args.lr)
            training_logs = []
            testing_logs=[]
            
            auc_sum_fold=[]
            aupr_sum_fold=[]
            f1_sum_fold=[]
            bacc_sum_fold=[]


            for step in range(1, args.epochs+1):
               
                log=train(args,Context_MiT4SL,train_loader, optimizer_model,train_data,num_train_node,ori_train_data,device)
           
                training_logs.append(log)
               
                
                # if args.do_test:
                eval_auc,eval_aupr,eval_f1,eval_bacc,testing_log=eval(args,Context_MiT4SL,test_loader,test_data,num_test_node,ori_test_data,device)
                auc_sum_fold.append(eval_auc)
                aupr_sum_fold.append(eval_aupr)
                f1_sum_fold.append(eval_f1)
                bacc_sum_fold.append(eval_bacc)
                testing_logs.append(testing_log)

                
            
                #store log information
                if step % args.log_steps == 0:
                    training_metrics = {}
                    testing_metrics={}
                   
                    for metric in training_logs[0].keys():
                        training_metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                    logging.info('loss on training Dataset...')
                    logging.info('eval on testing Dataset...')
                    log_metrics('Training average', step, training_metrics)
                    for metric in testing_logs[0].keys():
                        testing_metrics[metric] = sum([log[metric] for log in testing_logs])/len(testing_logs)
                    
                    log_metrics('Testing average', step, testing_metrics)

                    
                    training_logs = []
                    testing_logs=[]
                 

                if step % args.save_checkpoint_steps == 0: # save model
                    save_variable_list = {
                        'step': step, 
                        'current_learning_rate': args.lr,
                        'act':args.act,
                        'Data_used':args.Data_used,
                    }
                    save_model(Context_MiT4SL, optimizer_model,save_variable_list, args)

       # Save model
        save_variable_list = {
            'step': step, 
            'current_learning_rate':  args.lr,
            
        }
        save_model(Context_MiT4SL, optimizer_model, save_variable_list, args)
        print(f'fold_{n_fold}_auc:{round(auc_sum_fold[-1],4)}')
        print(f'fold_{n_fold}_aupr:{round(aupr_sum_fold[-1],4)}')
        print(f'fold_{n_fold}_f1:{round(f1_sum_fold[-1],4)}')
        print(f'fold_{n_fold}_bacc:{round(bacc_sum_fold[-1],4)}')
        


        # store the result
        eval_metric_folds['fold'].append(n_fold)
        eval_metric_folds['auc'].append(round(auc_sum_fold[-1],4))   
        eval_metric_folds['aupr'].append(round(aupr_sum_fold[-1],4)) 
        eval_metric_folds['f1'].append(round(f1_sum_fold[-1],4)) 
        eval_metric_folds['bacc'].append(round(bacc_sum_fold[-1],4)) 
    
        
    eval_metric_folds=pd.DataFrame(eval_metric_folds)
    eval_metric_folds.loc[5,'fold']='average'
    eval_metric_folds.loc[5,'auc']=round(eval_metric_folds['auc'].mean(),4)
    eval_metric_folds.loc[5,'aupr']=round(eval_metric_folds['aupr'].mean(),4)
    eval_metric_folds.loc[5,'f1']=round(eval_metric_folds['f1'].mean(),4)
    eval_metric_folds.loc[5,'bacc']=round(eval_metric_folds['bacc'].mean(),4)
    
    eval_metric_folds.loc[6,'fold']='std'
    eval_metric_folds.loc[6,'auc']=round(eval_metric_folds.loc[:5,'auc'].std(),4)
    eval_metric_folds.loc[6,'aupr']=round(eval_metric_folds.loc[:5,'aupr'].std(),4)
    eval_metric_folds.loc[6,'f1']=round(eval_metric_folds.loc[:5,'f1'].std(),4)
    eval_metric_folds.loc[6,'bacc']=round(eval_metric_folds.loc[:5,'bacc'].std(),4)
    
    
    if args.do_low_data:
        eval_metric_folds.to_csv(args.Save_model_path+'/'+args.cv+'_'+str(args.train_data_ratio)+'_'+str(args.lr)+'_'+str(args.epochs)+'_result_eval.csv',index=False)
    else:
        eval_metric_folds.to_csv(args.Save_model_path+'/'+args.cv+'_'+str(args.lr)+'_'+str(args.epochs)+'_result_eval.csv',index=False)
    

            
if __name__ == "__main__":
    s=time()
    main()
    e=time()
    print(f"Total running time: {round(e - s, 2)}s")



