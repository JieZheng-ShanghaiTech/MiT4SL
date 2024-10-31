import argparse
from time import time
import torch_geometric.transforms as T
from torch.optim import lr_scheduler
import torch
from configs import get_cfg_defaults
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models import *
import pandas as pd
from util import *
import pickle
import math
import random
import torch.nn.init as init
import pickle
from info_nce import InfoNCE




def train(cfg,epoch,context_mit4sl, train_loader, optimizer_model,sldata,num_train_node,ori_train_data,device):

    criterion = nn.CrossEntropyLoss()
    optimizer_model=optimizer_model
    context_mit4sl.train()
    batch_size=num_train_node
    loss_sum,aucu_sum,aupr_sum,bacc_sum,loss_cl,loss_mse = 0,0,0,0,0,0
    infoloss=InfoNCE()
    mseloss = nn.MSELoss()

    scheduler_model=lr_scheduler.StepLR(optimizer_model,step_size=cfg.SCHEDULER.STEP_SIZE,gamma=cfg.SCHEDULER.GAMMA)
    
    for step,batch in enumerate(tqdm(train_loader, desc="Iteration")):
       
        batch = batch.to(device)
        batch_sl=sldata[sldata[3]==1]
        batch_nosl=sldata[sldata[3]==0]
        batch_nosl.reset_index(drop=True,inplace=True)
        nosl_idx=list(range(batch_nosl.shape[0]))
        np.random.seed(epoch+cfg.SOLVER.REPEAT_EXP_SEED)
        np.random.shuffle(nosl_idx)
        sampled_idx=np.random.choice(nosl_idx,size=batch_sl.shape[0]*cfg.SOLVER.BATCH_POS_NEG_RATIO,replace=True)
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
        
        _,tri_emb1,tri_emb2,prediction_result,avergae_prediction=context_mit4sl(sldata,batch,batch_size,ori_train_data)
        loss_cl=infoloss(tri_emb1,tri_emb2)
        predicted_labels = torch.argmax(avergae_prediction, dim=1)
        loss_mse=mseloss(predicted_labels.float(),all_prediction_label.float())
        all_prediction_result=prediction_result
        optimizer_model.zero_grad()

        loss = criterion(all_prediction_result,all_prediction_label.long())
        loss+=cfg.SOLVER.BETA1*loss_cl+cfg.SOLVER.BETA2*loss_mse
        all_prediction=torch.max(all_prediction_result.detach(),dim=1)[1]
        aucu,aupr,bacc=compute_accuracy(all_prediction_label,all_prediction,all_prediction_result)
        loss.backward()
        optimizer_model.step()
        scheduler_model.step()
        loss_sum += float(loss.cpu().item())
        aucu_sum+=float(aucu)
        aupr_sum+=float(aupr)
        bacc_sum+=float(bacc)
        # log = {
        #     'loss': loss_sum/(step+1),
        #     'auc':aucu_sum/(step+1),
        #     'aupr':aupr_sum/(step+1),
        #     'bacc':bacc_sum/(step+1), 
        # }
        log = {
            'loss': loss_sum/(step+1),
        }
   
    return log




def eval(cfg,context_mit4sl,val_loader,test_loader,valdata,sldata,num_val_node,num_test_node,ori_val_data,ori_test_data,device):
    context_mit4sl.eval()
   
    aucu_sum,aupr_sum,bacc_sum=0,0,0
    stop_counts = 0
    best_valid_aupr = 0
    best_test_aucu,best_test_aupr,best_test_bacc = 0,0,0
 
    with torch.no_grad():
        batch_size=num_val_node
        for step,batch in enumerate(val_loader):
            batch = batch.to(device)
            prediction_label=valdata[3]
            
            _,_,_,prediction_result,_=context_mit4sl(valdata,batch,batch_size,ori_val_data)
            all_prediction_label=prediction_label
            all_prediction_result=prediction_result
            all_prediction=torch.max(all_prediction_result.detach(),dim=1)[1]
            all_prediction_label=torch.tensor(all_prediction_label.values).to(device)
            valid_aucu,valid_aupr,valid_bacc=compute_accuracy(all_prediction_label,all_prediction,all_prediction_result)
            valid_log={
                'valid_aupr':valid_aupr, 
            }
    with torch.no_grad():
        batch_size=num_test_node
        for step,batch in enumerate(test_loader):
            batch = batch.to(device)
            prediction_label=sldata[3]
            _,_,_,prediction_result,_=context_mit4sl(sldata,batch,batch_size,ori_test_data)
            all_prediction_label=prediction_label
            all_prediction_result=prediction_result
            all_prediction=torch.max(all_prediction_result.detach(),dim=1)[1]
            all_prediction_label=torch.tensor(all_prediction_label.values).to(device)
            aucu,aupr,bacc=compute_accuracy(all_prediction_label,all_prediction,all_prediction_result)
        
            if valid_aupr > best_valid_aupr:
                aucu_sum+=float(aucu)
                aupr_sum+=float(aupr)
                bacc_sum+=float(bacc)
                best_valid_aupr = valid_aupr
                best_test_aucu=aucu_sum/(step+1)
                best_test_aupr = aupr_sum/(step+1)
                best_test_bacc=bacc_sum/(step+1)
                log = {
                'auc':best_test_aucu,
                'aupr':best_test_aupr,
                'bacc':best_test_bacc,
            }
                stop_counts = 0
            else:
                stop_counts += 1
            if (stop_counts == cfg.SOLVER.STOP_COUNTS):
                print('Early stopped.')
                break
                
            
      
        return best_test_aucu,best_test_aupr,best_test_bacc,valid_log,log
        
        


def main():
    parser=argparse.ArgumentParser(description='MiT4SL for cell line SL prediction')
    parser.add_argument('--cfg_path',help='path to config file',type=str,default='./configs/MiT4SL_adapted_Multi_5_to_A549.yaml')
    parser.add_argument('--Save_model_path',help='path to config file',type=str,default=None)
    args=parser.parse_args()
    cfg=get_cfg_defaults()
    cfg.merge_from_file(args.cfg_path)
    
    torch.manual_seed(0)
    np.random.seed(0)
    set_seed(0)

    # check the input data 
    if cfg.SOLVER.TASK_DATAPATH is None: 
        raise ValueError('one of init_checkpoint/data_path must be choosed.')
    # construct the result store path
    if cfg.SOLVER.USE_DATA=='KG_Seq_Cell_Line':
        args.Save_model_path=cfg.RESULT.SAVE_PATH+'/'+cfg.SOLVER.SCENARIO+'/'+cfg.SOLVER.CELL+'/'+cfg.SOLVER.USE_DATA+'_'+str(cfg.SOLVER.BETA1)+'_'+str(cfg.SOLVER.BETA2)+'_'+str(cfg.SOLVER.EPOCHS)          
        if args.Save_model_path and not os.path.exists(args.Save_model_path): 
            os.makedirs(args.Save_model_path)
    else:
        raise ValueError('one of KG&Seq mode must be choosed.')

    device = torch.device("cuda:" + str(cfg.SOLVER.DEVICE)) if torch.cuda.is_available() else torch.device("cpu")
    set_logger(args)
    # Load and initialize the feature data
    with open(cfg.SOLVER.KG_NODE_DICT,'rb') as f:
        node_index=json.load(f)

    gene_protein=node_index[cfg.SOLVER.NODE_TYPE]
    node_type=cfg.SOLVER.NODE_TYPE
    kgdata,cell_ppidata=init_graph_data(cfg.SOLVER.KG_DATAPATH,cfg.SOLVER.CELLNX_DATAPATH)
    new_proteinseq_data,cell_line_proteins=overlapping_with_sequence(cfg.SOLVER.PROTEINSeq_DATAPATH,cfg.SOLVER.CELLPROTEIN_DATAPATH,cfg.SOLVER.TASK_DATAPATH,cfg.SOLVER.CELL)

    logging.info('Model: MiT4SL')
    logging.info('KG data: %s' % cfg.KG.NAME) 
    logging.info('Seq data: %s' % cfg.ProteinSeq.NAME) 
    logging.info(f'Cell_line_proteins: Multi_cell_{cfg.SOLVER.CELL}' ) 
    logging.info('Data_used: %s' %cfg.SOLVER.USE_DATA)
    
    # initiliaze the model
    result_text=''
    result_text+=f"""
                           MiT4SL
                        {cfg.SOLVER.CELL}
                    ----------------------------
    """
    eval_metric_folds={'fold':[],'auc':[],'aupr':[],'bacc':[]}
    for i in range(5):
        logging.info(f'Fold_{i} training...')
        n_fold=i
        #set up models
        context_mit4sl=MiT4SL(kgdata,cell_ppidata,new_proteinseq_data,cell_line_proteins,
                            cfg.KG.HIDEEN_DIM, cfg.KG.EMB_DIM, 
                            cfg.KG.NUM_HEADS,  cfg.KG.NUM_LAYERS,
                            cfg.Cell_Line.HIDDEN_DIM, cfg.Cell_Line.NUM_LAYERS,
                            cfg.KG.USE_KG,cfg.Cell_Line.USE_Cell,cfg.ProteinSeq.USE_Seq,device)
        #set up optimizers
        optimizer_model = optim.Adam(context_mit4sl.parameters(), lr=cfg.SOLVER.LR, weight_decay=0)
    
  
        logging.info('Ramdomly Initializing Context_LukePi Model...')  
        init_step = 0
        step = init_step 
        
    
        train_data,val_data,test_data,train_mask,val_mask,test_mask,num_train_node,num_val_node,num_test_node,ori_train_data,ori_val_data,ori_test_data=Downstream_data_preprocess_cell(cfg.SOLVER.TASK_DATAPATH,cfg.SOLVER.CELL,gene_protein)

        train_loader,val_loader,test_loader=Construct_loader(cfg.KG_SAMPLER.SAMPLE_NODES,cfg.KG_SAMPLER.SAMPLE_LAYERS,cfg.SOLVER.NUM_WORKERS,
                                                            kgdata,train_mask,val_mask,test_mask,node_type,
                                                            num_train_node,num_val_node,num_test_node)
        logging.info('Start Training...')
        logging.info('init_step = %d' % init_step)
        logging.info('num_train_node = %d' % num_train_node)
        logging.info('num_test_node = %d' % num_test_node)
        logging.info('learning_rate = %f' %cfg.SOLVER.LR)
        training_logs,val_logs,testing_logs = [],[],[]
        auc_sum_fold,aupr_sum_fold,bacc_sum_fold=[],[],[]
 
        for step in range(1, cfg.SOLVER.EPOCHS+1):
            
            log=train(cfg,step,context_mit4sl,train_loader, optimizer_model,train_data,num_train_node,ori_train_data,device)
            training_logs.append(log)
            # test
            eval_auc,eval_aupr,eval_bacc,valid_log,testing_log=eval(cfg,context_mit4sl,val_loader,test_loader,
                                                                val_data,test_data,num_val_node,num_test_node,
                                                                ori_val_data,ori_test_data,device)
            auc_sum_fold.append(eval_auc)
            aupr_sum_fold.append(eval_aupr)
            bacc_sum_fold.append(eval_bacc)
            testing_logs.append(testing_log)
            val_logs.append(valid_log)
            #store log information
            if step % cfg.RESULT.LOG_STEPS == 0:
                training_metrics = {}
                testing_metrics={}
                val_metrics={}
                
                for metric in training_logs[0].keys():
                    training_metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                for metric in val_logs[0].keys():
                    val_metrics[metric] = sum([log[metric] for log in val_logs])/len(val_logs)
                logging.info('============= Start Training ... ==============')
                log_metrics('Training average', step, training_metrics)
                logging.info('============= Start Validating...==============')
                log_metrics('Valid average', step, val_metrics)
                for metric in testing_logs[0].keys():
                    testing_metrics[metric] = sum([log[metric] for log in testing_logs])/len(testing_logs)
                logging.info('============= Start Testing ... ===============')
                log_metrics('Testing average', step, testing_metrics)

                
                training_logs = []
                testing_logs=[]
                

            if step % cfg.RESULT.SAVE_CHEACKPOINTS_STEP == 0: # save model
                save_model(context_mit4sl, optimizer_model,args.Save_model_path)

       # Save model
 
        save_model(context_mit4sl, optimizer_model,args.Save_model_path)
        print(f'fold_{n_fold}_auc:{round(auc_sum_fold[-1],4)}')
        print(f'fold_{n_fold}_aupr:{round(aupr_sum_fold[-1],4)}')
        print(f'fold_{n_fold}_bacc:{round(bacc_sum_fold[-1],4)}')
        # store the result
        eval_metric_folds['fold'].append(n_fold)
        eval_metric_folds['auc'].append(round(auc_sum_fold[-1],4))   
        eval_metric_folds['aupr'].append(round(aupr_sum_fold[-1],4)) 
        eval_metric_folds['bacc'].append(round(bacc_sum_fold[-1],4)) 
        result_text+=f"""
                    
                                fold_{n_fold}
                        ----------------------------     
                        AUC:{round(auc_sum_fold[-1],4)}
                        AUPR:{round(aupr_sum_fold[-1],4)}
                        BACC:{round(bacc_sum_fold[-1],4)}
                        ----------------------------
                    """  
        
        # eval_metric_folds['precision'].append(round(precision_sum_fold[-1],4)) 
        
    eval_metric_folds=pd.DataFrame(eval_metric_folds)
    eval_metric_folds.loc[5,'fold']='average'
    eval_metric_folds.loc[5,'auc']=round(eval_metric_folds['auc'].mean(),4)
    eval_metric_folds.loc[5,'aupr']=round(eval_metric_folds['aupr'].mean(),4)
    eval_metric_folds.loc[5,'bacc']=round(eval_metric_folds['bacc'].mean(),4)
    eval_metric_folds.loc[6,'fold']='std'
    eval_metric_folds.loc[6,'auc']=round(eval_metric_folds.loc[:5,'auc'].std(),4)
    eval_metric_folds.loc[6,'aupr']=round(eval_metric_folds.loc[:5,'aupr'].std(),4)
    eval_metric_folds.loc[6,'bacc']=round(eval_metric_folds.loc[:5,'bacc'].std(),4)
    result_text+=f"""       
                        ----------------------------
                        AUC_mean(std):{round(eval_metric_folds.loc[5,'auc'],4)}({round(eval_metric_folds.loc[6,'auc'],4)})
                        AUPR_mean(std):{round(eval_metric_folds.loc[5,'aupr'],4)}({round(eval_metric_folds.loc[6,'aupr'],4)})
                        BACC_mean(std):{round(eval_metric_folds.loc[5,'bacc'].mean(),4)}({round(eval_metric_folds.loc[6,'bacc'],4)})
                        ----------------------------
                        """ 
    save_root=args.Save_model_path
    os.makedirs(save_root,exist_ok=True)
    with open(os.path.join(save_root,f"{cfg.SOLVER.CELL}_results.txt"),'w') as f:
        f.write(result_text)
    eval_metric_folds.to_csv(args.Save_model_path+'/'+'final_result_eval.csv',index=False)
    

            
if __name__ == "__main__":
    s=time()
    main()
    e=time()
    print(f"Total running time: {round(e - s, 2)}s")



