import torch
import torch.nn.functional as F
# from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear, ReLU
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DeepGCNLayer, GENConv,HGTConv
import pandas as pd
import numpy as np

class ContextMiT4SL(torch.nn.Module):
    def __init__(self,kgdata,cell_ppidata,proteinseq_data,cell_line_proteins,
                 KG_hidden_channels,KG_out_channels, KG_num_heads, KG_num_layers,
                 CellLineGraph_hidden_channels, CellLineGraph_num_layers,
                 Use_kg,Use_cellnx,Use_seq,device,exdata=None,cndata=None,ex_protein=None,cn_protein=None,new_cl=None,Use_omics=None):
        super().__init__()
        self.emb_dim=KG_out_channels
        self.Use_kg=Use_kg
        self.Use_cellnx=Use_cellnx
        self.Use_seq=Use_seq
        self.Use_omics=Use_omics
        self.cell_line_proteins=cell_line_proteins
        
       # self.Use_combine=Use_combine
       # self.Use_cl=Use_cl
       
        self.kgdata=kgdata
        self.cell_ppidata=cell_ppidata
        self.proteinseq_data=proteinseq_data
        self.exdata=exdata
        self.cndata=cndata
        self.new_cl=new_cl
        self.ex_protein=ex_protein
        self.cn_protein=cn_protein
        self.device=device
        self.linear_combine=torch.nn.Linear(2*self.emb_dim,2*self.emb_dim).to(device)
        self.Linear_classifier_both=torch.nn.Sequential(torch.nn.Linear(self.emb_dim*6,self.emb_dim),torch.nn.ReLU(),torch.nn.Linear(self.emb_dim,2)).to(device)
        self.Linear_classifier_both_ablation=torch.nn.Sequential(torch.nn.Linear(self.emb_dim*5,self.emb_dim),torch.nn.ReLU(),torch.nn.Linear(self.emb_dim,2)).to(device)
        self.Linear_classifier_singleall=torch.nn.Sequential(torch.nn.Linear(self.emb_dim*4,self.emb_dim),torch.nn.ReLU(),torch.nn.Linear(self.emb_dim,2)).to(device)
        #self.Linear_classifier_singleseq=torch.nn.Sequential(torch.nn.Linear(self.emb_dim*5,self.emb_dim),torch.nn.ReLU(),torch.nn.Linear(self.emb_dim,2)).to(device)
        self.Linear_classifier_single=torch.nn.Sequential(torch.nn.Linear(self.emb_dim*4,self.emb_dim),torch.nn.ReLU(),torch.nn.Linear(self.emb_dim,2)).to(device)
        self.Linear_classifier_single_ablation=torch.nn.Sequential(torch.nn.Linear(self.emb_dim*3,self.emb_dim),torch.nn.ReLU(),torch.nn.Linear(self.emb_dim,2)).to(device)
        self.Linear_classifier_omics=torch.nn.Sequential(torch.nn.Linear(self.emb_dim*6,self.emb_dim),torch.nn.ReLU(),torch.nn.Linear(self.emb_dim,2)).to(device)
        self.Linear_classifier_all=torch.nn.Sequential(torch.nn.Linear(384,2*self.emb_dim),torch.nn.ReLU(),torch.nn.Linear(2*self.emb_dim,self.emb_dim),torch.nn.ReLU(),torch.nn.Linear(self.emb_dim,2)).to(device)
        
        if Use_kg:
            self.KG_encoder=KGEncoder(kgdata,KG_hidden_channels,KG_out_channels, KG_num_heads, KG_num_layers).to(device)
        else:
            print('Do not use KG encoder.')
        if Use_cellnx:
            self.CellGraph_encoder=CellLineGraphEncoder(cell_ppidata,CellLineGraph_hidden_channels,self.emb_dim,CellLineGraph_num_layers).to(device)
           
        else:
            print('Do not use CellGraph encoder.')
        if Use_seq:
            self.Sequence_encoder=SequenceEncoder(640,64).to(device)
        else:
            print('Do not use Sequence encoder.')
        if Use_omics:
            self.Omics_encoder=OmicsEncoder(self.emb_dim,).to(device)
        else:
            print('Do not use Omics encoder.')
        

    def forward(self,sldata,act_operation,batch,batch_size,ori_data): 
        nodea,nodeb=ori_data[0],ori_data[1]
        cell_id=ori_data[2]

        if self.Use_kg:
            kgemb_a,kgemb_b=self.KG_encoder(batch.x_dict,batch.edge_index_dict,batch,batch_size,sldata)
           
        if self.Use_seq:
            sequence_emb,cellseq_emb=self.Sequence_encoder(self.cell_line_proteins,self.proteinseq_data,self.device)
            seqemb_a=torch.stack([sequence_emb[i] for i in nodea.values])
            seqemb_b=torch.stack([sequence_emb[i] for i in nodeb.values])

        if self.Use_omics:
            omics_emb=self.Omics_encoder(self.new_cl,self.device)
            # omicsemb_a=torch.stack([omics_emb[i] for i in nodea.values])
            # omicsemb_b=torch.stack([omics_emb[i] for i in nodeb.values])


        if self.Use_cellnx:
            cellgraph_emb= self.CellGraph_encoder(self.cell_ppidata,self.device)
            cellgraph_emb=torch.stack(cellgraph_emb).squeeze(1)

        if act_operation=='KGSeq_combineBoth':
            final_emba=torch.cat((kgemb_a,seqemb_a),dim=1)
            final_embb=torch.cat((kgemb_b,seqemb_b),dim=1)
            edge_emb=torch.cat((final_emba,final_embb),dim=1)
            #--------concate embeddings from two modals-------
            cellemb_nx=cellgraph_emb[cell_id.values].squeeze(1)
            cellemb_seq=cellseq_emb[cell_id.values].squeeze(1)
            cell_emb=torch.cat((cellemb_nx,cellemb_seq),dim=1)
            tri_emb=torch.cat((edge_emb,cell_emb),dim=1)
            prediction=self.Linear_classifier_both(tri_emb)
        elif act_operation=='KGSeq_CL':
            final_emba=torch.cat((kgemb_a,seqemb_a),dim=1)
            final_emba=self.linear_combine(final_emba)
            final_embb=torch.cat((kgemb_b,seqemb_b),dim=1)
            final_embb=self.linear_combine(final_embb)
            edge_emb=torch.cat((final_emba,final_embb),dim=1)
            #--------contrastive learning-------
            cellemb_nx=cellgraph_emb[cell_id.values].squeeze(1)
            cellemb_seq=cellseq_emb[cell_id.values].squeeze(1)
            cell_emb=torch.cat((cellemb_nx,cellemb_seq),dim=1)
            tri_emb=torch.cat((edge_emb,cell_emb),dim=1)
            prediction=self.Linear_classifier_both(tri_emb)
            return tri_emb,prediction
        
        elif act_operation=='KGSeq_multiCL':
            cellemb_nx=cellgraph_emb[cell_id.values].squeeze(1)
            cellemb_seq=cellseq_emb[cell_id.values].squeeze(1)
            cell_emb=torch.cat((cellemb_nx,cellemb_seq),dim=1)
            final_emba=torch.cat((kgemb_a,seqemb_a),dim=1)
            final_embb=torch.cat((kgemb_b,seqemb_b),dim=1)
            edge_emb=torch.cat((final_emba,final_embb),dim=1)
            tri_emb=torch.cat((edge_emb,cell_emb),dim=1)
            prediction=self.Linear_classifier_both(tri_emb)
            kgemb=torch.cat((kgemb_a,kgemb_b),dim=1)
            kgemb=torch.cat((kgemb,cell_emb),dim=1)
            seqemb=torch.cat((seqemb_a,seqemb_b),dim=1)
            seqemb=torch.cat((seqemb,cell_emb),dim=1)
            prediction_kg=self.Linear_classifier_single(kgemb)
            prediction_seq=self.Linear_classifier_single(seqemb)
            avergae_prediction=torch.sigmoid((prediction_kg+prediction_seq)/2)
            return tri_emb,prediction,avergae_prediction


        elif act_operation=='KGSeq_finalCL':
            cellemb_nx=cellgraph_emb[cell_id.values].squeeze(1)
            cellemb_seq=cellseq_emb[cell_id.values].squeeze(1)
            cell_emb=torch.cat((cellemb_nx,cellemb_seq),dim=1)
            # ---------CL for two sample pairs--------
            nodea_emb=torch.stack((kgemb_a,seqemb_a),dim=0)
            nodeb_emb=torch.stack((kgemb_b,seqemb_b),dim=0)
            sample_a_indices1 = torch.randint(0, 2, (1,))
            sample_b_indices1 = torch.randint(0, 2, (1,))
            final_emba1=nodea_emb[sample_a_indices1].squeeze(0)
            final_embb1=nodeb_emb[sample_b_indices1].squeeze(0)
            edge_emb_CL1=torch.cat((final_emba1,final_embb1),dim=1)
            tri_emb1=torch.cat((edge_emb_CL1,cell_emb),dim=1)
            sample_a_indices2 = torch.randint(0, 2, (1,))
            sample_b_indices2 = torch.randint(0, 2, (1,))
            final_emba2=nodea_emb[sample_a_indices2].squeeze(0)
            final_embb2=nodeb_emb[sample_b_indices2].squeeze(0)
            edge_emb_CL2=torch.cat((final_emba2,final_embb2),dim=1)
            tri_emb2=torch.cat((edge_emb_CL2,cell_emb),dim=1)
            final_emba=torch.cat((kgemb_a,seqemb_a),dim=1)
            final_embb=torch.cat((kgemb_b,seqemb_b),dim=1)
            edge_emb=torch.cat((final_emba,final_embb),dim=1)
            tri_emb=torch.cat((edge_emb,cell_emb),dim=1)
            prediction=self.Linear_classifier_both(tri_emb)
            kgemb=torch.cat((kgemb_a,kgemb_b,cell_emb),dim=1)
            seqemb=torch.cat((seqemb_a,seqemb_b,cell_emb),dim=1)
            prediction_kg=self.Linear_classifier_single(kgemb)
            prediction_seq=self.Linear_classifier_single(seqemb)
            avergae_prediction=torch.sigmoid((prediction_kg+prediction_seq)/2)
            return tri_emb1,tri_emb2,prediction,avergae_prediction
        


        elif act_operation=='KGSeq_finalCL_ppi':
            cellemb_nx=cellgraph_emb[cell_id.values].squeeze(1)
            # cellemb_seq=cellseq_emb[cell_id.values].squeeze(1)
            cell_emb=cellemb_nx
            # ---------CL for two sample pairs--------
            nodea_emb=torch.stack((kgemb_a,seqemb_a),dim=0)
            nodeb_emb=torch.stack((kgemb_b,seqemb_b),dim=0)
            sample_a_indices1 = torch.randint(0, 2, (1,))
            sample_b_indices1 = torch.randint(0, 2, (1,))
            final_emba1=nodea_emb[sample_a_indices1].squeeze(0)
            final_embb1=nodeb_emb[sample_b_indices1].squeeze(0)
            edge_emb_CL1=torch.cat((final_emba1,final_embb1),dim=1)
            tri_emb1=torch.cat((edge_emb_CL1,cell_emb),dim=1)
            sample_a_indices2 = torch.randint(0, 2, (1,))
            sample_b_indices2 = torch.randint(0, 2, (1,))
            final_emba2=nodea_emb[sample_a_indices2].squeeze(0)
            final_embb2=nodeb_emb[sample_b_indices2].squeeze(0)
            edge_emb_CL2=torch.cat((final_emba2,final_embb2),dim=1)
            tri_emb2=torch.cat((edge_emb_CL2,cell_emb),dim=1)
            final_emba=torch.cat((kgemb_a,seqemb_a),dim=1)
            final_embb=torch.cat((kgemb_b,seqemb_b),dim=1)
            edge_emb=torch.cat((final_emba,final_embb),dim=1)
            tri_emb=torch.cat((edge_emb,cell_emb),dim=1)
            prediction=self.Linear_classifier_both_ablation(tri_emb)
            kgemb=torch.cat((kgemb_a,kgemb_b,cell_emb),dim=1)
            seqemb=torch.cat((seqemb_a,seqemb_b,cell_emb),dim=1)
            prediction_kg=self.Linear_classifier_single_ablation(kgemb)
            prediction_seq=self.Linear_classifier_single_ablation(seqemb)
            avergae_prediction=torch.sigmoid((prediction_kg+prediction_seq)/2)
            return tri_emb1,tri_emb2,prediction,avergae_prediction
        
        elif act_operation=='KGSeq_finalCL_seq':
            # cellemb_nx=cellgraph_emb[cell_id.values].squeeze(1)
            cellemb_seq=cellseq_emb[cell_id.values].squeeze(1)
            # cell_emb=torch.cat((cellemb_nx,cellemb_seq),dim=1)
            cell_emb=cellemb_seq
            # ---------CL for two sample pairs--------
            nodea_emb=torch.stack((kgemb_a,seqemb_a),dim=0)
            nodeb_emb=torch.stack((kgemb_b,seqemb_b),dim=0)
            sample_a_indices1 = torch.randint(0, 2, (1,))
            sample_b_indices1 = torch.randint(0, 2, (1,))
            final_emba1=nodea_emb[sample_a_indices1].squeeze(0)
            final_embb1=nodeb_emb[sample_b_indices1].squeeze(0)
            edge_emb_CL1=torch.cat((final_emba1,final_embb1),dim=1)
            tri_emb1=torch.cat((edge_emb_CL1,cell_emb),dim=1)
            sample_a_indices2 = torch.randint(0, 2, (1,))
            sample_b_indices2 = torch.randint(0, 2, (1,))
            final_emba2=nodea_emb[sample_a_indices2].squeeze(0)
            final_embb2=nodeb_emb[sample_b_indices2].squeeze(0)
            edge_emb_CL2=torch.cat((final_emba2,final_embb2),dim=1)
            tri_emb2=torch.cat((edge_emb_CL2,cell_emb),dim=1)
            final_emba=torch.cat((kgemb_a,seqemb_a),dim=1)
            final_embb=torch.cat((kgemb_b,seqemb_b),dim=1)
            edge_emb=torch.cat((final_emba,final_embb),dim=1)
            tri_emb=torch.cat((edge_emb,cell_emb),dim=1)
            prediction=self.Linear_classifier_both_ablation(tri_emb)
            kgemb=torch.cat((kgemb_a,kgemb_b,cell_emb),dim=1)
            seqemb=torch.cat((seqemb_a,seqemb_b,cell_emb),dim=1)
            prediction_kg=self.Linear_classifier_single_ablation(kgemb)
            prediction_seq=self.Linear_classifier_single_ablation(seqemb)
            avergae_prediction=torch.sigmoid((prediction_kg+prediction_seq)/2)
            return tri_emb1,tri_emb2,prediction,avergae_prediction

        elif act_operation=='KGSeq_finalCL_KG':
            cellemb_nx=cellgraph_emb[cell_id.values].squeeze(1)
            cellemb_seq=cellseq_emb[cell_id.values].squeeze(1)
            cell_emb=torch.cat((cellemb_nx,cellemb_seq),dim=1)
            # ---------CL for two sample pairs--------
            nodea_emb=torch.stack((kgemb_a,seqemb_a),dim=0)
            nodeb_emb=torch.stack((kgemb_b,seqemb_b),dim=0)
            sample_a_indices1 = torch.randint(0, 2, (1,))
            sample_b_indices1 = torch.randint(0, 2, (1,))
            final_emba1=nodea_emb[sample_a_indices1].squeeze(0)
            final_embb1=nodeb_emb[sample_b_indices1].squeeze(0)
            edge_emb_CL1=torch.cat((final_emba1,final_embb1),dim=1)
            tri_emb1=torch.cat((edge_emb_CL1,cell_emb),dim=1)
            sample_a_indices2 = torch.randint(0, 2, (1,))
            sample_b_indices2 = torch.randint(0, 2, (1,))
            final_emba2=nodea_emb[sample_a_indices2].squeeze(0)
            final_embb2=nodeb_emb[sample_b_indices2].squeeze(0)
            edge_emb_CL2=torch.cat((final_emba2,final_embb2),dim=1)
            tri_emb2=torch.cat((edge_emb_CL2,cell_emb),dim=1)
            final_emba=torch.cat((kgemb_a,seqemb_a),dim=1)
            final_embb=torch.cat((kgemb_b,seqemb_b),dim=1)
            edge_emb=torch.cat((final_emba,final_embb),dim=1)
            tri_emb=torch.cat((edge_emb,cell_emb),dim=1)
            prediction=self.Linear_classifier_both(tri_emb)
            kgemb=torch.cat((kgemb_a,kgemb_b,cell_emb),dim=1)
            seqemb=torch.cat((seqemb_a,seqemb_b,cell_emb),dim=1)
            prediction_kg=self.Linear_classifier_single(kgemb)
            prediction_seq=self.Linear_classifier_single(seqemb)
            avergae_prediction=torch.sigmoid((prediction_kg+prediction_seq)/2)
            return tri_emb1,tri_emb2,prediction,avergae_prediction





        

        elif act_operation=='KGSeq_finalCL_ex':
            cellemb_nx=cellgraph_emb[cell_id.values].squeeze(1)
            cellemb_seq=cellseq_emb[cell_id.values].squeeze(1)
            cell_emb=torch.cat((cellemb_nx,cellemb_seq),dim=1)
            # ---------CL for two sample pairs--------
            nodea_emb=torch.stack((kgemb_a,seqemb_a),dim=0)
            nodeb_emb=torch.stack((kgemb_b,seqemb_b),dim=0)
            sample_a_indices1 = torch.randint(0, 2, (1,))
            sample_b_indices1 = torch.randint(0, 2, (1,))
            final_emba1=nodea_emb[sample_a_indices1].squeeze(0)
            final_embb1=nodeb_emb[sample_b_indices1].squeeze(0)
            edge_emb_CL1=torch.cat((final_emba1,final_embb1),dim=1)
            tri_emb1=torch.cat((edge_emb_CL1,cell_emb),dim=1)
            sample_a_indices2 = torch.randint(0, 2, (1,))
            sample_b_indices2 = torch.randint(0, 2, (1,))
            final_emba2=nodea_emb[sample_a_indices2].squeeze(0)
            final_embb2=nodeb_emb[sample_b_indices2].squeeze(0)
            edge_emb_CL2=torch.cat((final_emba2,final_embb2),dim=1)
            tri_emb2=torch.cat((edge_emb_CL2,cell_emb),dim=1)
            final_emba=torch.cat((kgemb_a,seqemb_a),dim=1)
            final_embb=torch.cat((kgemb_b,seqemb_b),dim=1)
            edge_emb=torch.cat((final_emba,final_embb),dim=1)
            tri_emb=torch.cat((edge_emb,cell_emb),dim=1)
            prediction=self.Linear_classifier_both(tri_emb)
            kgemb=torch.cat((kgemb_a,kgemb_b,cell_emb),dim=1)
            seqemb=torch.cat((seqemb_a,seqemb_b,cell_emb),dim=1)
            prediction_kg=self.Linear_classifier_single(kgemb)
            prediction_seq=self.Linear_classifier_single(seqemb)
            avergae_prediction=torch.sigmoid((prediction_kg+prediction_seq)/2)
            return tri_emb1,tri_emb2,prediction,avergae_prediction


        elif act_operation=='KGSeq_final_omics_CL':
            # cellemb_nx=cellgraph_emb[cell_id.values].squeeze(1)
            # cellemb_seq=cellseq_emb[cell_id.values].squeeze(1)
            #cellemb_omics=cellemb_omics[cell_id.values].squeeze(1)
           # cell_emb=torch.cat((cellemb_nx,cellemb_seq,cellemb_omics),dim=1)
            cell_emb=omics_emb[cell_id.values]
            # ---------CL for two sample pairs--------
            nodea_emb=torch.stack((kgemb_a,seqemb_a),dim=0)
            nodeb_emb=torch.stack((kgemb_b,seqemb_b),dim=0)
            # sample_a_indices1 = torch.randint(0, 3, (1,))
            # sample_b_indices1 = torch.randint(0, 3, (1,))
            sample_a_indices1 = torch.randint(0, 2, (1,))
            sample_b_indices1 = torch.randint(0, 2, (1,))
            final_emba1=nodea_emb[sample_a_indices1].squeeze(0)
            final_embb1=nodeb_emb[sample_b_indices1].squeeze(0)

            # final_emba1=torch.cat((final_emba1,omicsemb_a),dim=1)
            # final_embb1=torch.cat((final_embb1,omicsemb_b),dim=1)

            edge_emb_CL1=torch.cat((final_emba1,final_embb1),dim=1)
           
            tri_emb1=torch.cat((edge_emb_CL1,cell_emb),dim=1)

       
            sample_a_indices2 = torch.randint(0, 2, (1,))
            sample_b_indices2 = torch.randint(0, 2, (1,))
            final_emba2=nodea_emb[sample_a_indices2].squeeze(0)
            final_embb2=nodeb_emb[sample_b_indices2].squeeze(0)
       
            
            edge_emb_CL2=torch.cat((final_emba2,final_embb2),dim=1)
            tri_emb2=torch.cat((edge_emb_CL2,cell_emb),dim=1)

            final_emba=torch.cat((kgemb_a,seqemb_a),dim=1)
            final_embb=torch.cat((kgemb_b,seqemb_b),dim=1)
            edge_emb=torch.cat((final_emba,final_embb),dim=1)
            tri_emb=torch.cat((edge_emb,cell_emb),dim=1)
            prediction=self.Linear_classifier_both(tri_emb)
            kgemb=torch.cat((kgemb_a,kgemb_b,cell_emb),dim=1)
            seqemb=torch.cat((seqemb_a,seqemb_b,cell_emb),dim=1)
            #omicsemb=torch.cat((omicsemb_a,omicsemb_b,cell_emb),dim=1)
            prediction_kg=self.Linear_classifier_singleall(kgemb)
            prediction_seq=self.Linear_classifier_singleall(seqemb)
       
            avergae_prediction=torch.sigmoid((prediction_kg+prediction_seq)/2)
            return tri_emb1,tri_emb2,prediction,avergae_prediction


        elif act_operation=='KG_only':
            final_emb=torch.cat((kgemb_a,kgemb_b),dim=1)
            cellemb_nx=cellgraph_emb[cell_id.values].squeeze(1)
            cellemb_seq=cellseq_emb[cell_id.values].squeeze(1)
            cell_emb=torch.cat((cellemb_nx,cellemb_seq),dim=1)
            tri_emb=torch.cat((final_emb,cell_emb),dim=1)
            prediction=self.Linear_classifier_single(tri_emb)   
        elif act_operation=='Seq_only':
            final_emb=torch.cat((seqemb_a,seqemb_b),dim=1)
            cellemb_nx=cellgraph_emb[cell_id.values].squeeze(1)
            cellemb_seq=cellseq_emb[cell_id.values].squeeze(1)
            cell_emb=torch.cat((cellemb_nx,cellemb_seq),dim=1)
            tri_emb=torch.cat((final_emb,cell_emb),dim=1)
            prediction=self.Linear_classifier_single(tri_emb)

        elif act_operation=='No_cell':
            # final_emba=torch.cat((kgemb_a,seqemb_a),dim=1)
            # final_embb=torch.cat((kgemb_b,seqemb_b),dim=1)
            # edge_emb=torch.cat((final_emba,final_embb),dim=1)
            # tri_emb=edge_emb
            # prediction=self.Linear_classifier_single(tri_emb)
            # cellemb_nx=cellgraph_emb[cell_id.values].squeeze(1)
            # cellemb_seq=cellseq_emb[cell_id.values].squeeze(1)
            # cell_emb=torch.cat((cellemb_nx,cellemb_seq),dim=1)
            # ---------CL for two sample pairs--------
            nodea_emb=torch.stack((kgemb_a,seqemb_a),dim=0)
            nodeb_emb=torch.stack((kgemb_b,seqemb_b),dim=0)
            sample_a_indices1 = torch.randint(0, 2, (1,))
            sample_b_indices1 = torch.randint(0, 2, (1,))
            final_emba1=nodea_emb[sample_a_indices1].squeeze(0)
            final_embb1=nodeb_emb[sample_b_indices1].squeeze(0)
            edge_emb_CL1=torch.cat((final_emba1,final_embb1),dim=1)
            tri_emb1=edge_emb_CL1
            sample_a_indices2 = torch.randint(0, 2, (1,))
            sample_b_indices2 = torch.randint(0, 2, (1,))
            final_emba2=nodea_emb[sample_a_indices2].squeeze(0)
            final_embb2=nodeb_emb[sample_b_indices2].squeeze(0)
            edge_emb_CL2=torch.cat((final_emba2,final_embb2),dim=1)
            # tri_emb2=torch.cat((edge_emb_CL2,cell_emb),dim=1)
            tri_emb2=edge_emb_CL2
            final_emba=torch.cat((kgemb_a,seqemb_a),dim=1)
            final_embb=torch.cat((kgemb_b,seqemb_b),dim=1)
            edge_emb=torch.cat((final_emba,final_embb),dim=1)
            tri_emb=edge_emb
            prediction=self.Linear_classifier_both_ablation(tri_emb)
            kgemb=torch.cat((kgemb_a,kgemb_b),dim=1)
            seqemb=torch.cat((seqemb_a,seqemb_b),dim=1)
            prediction_kg=self.Linear_classifier_single_ablation(kgemb)
            prediction_seq=self.Linear_classifier_single_ablation(seqemb)
            avergae_prediction=torch.sigmoid((prediction_kg+prediction_seq)/2)

            return tri_emb1,tri_emb2,prediction,avergae_prediction
        
        return prediction


        


class KGEncoder(torch.nn.Module):
    def __init__(self, data,hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(16, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        self.linproteins = Linear(hidden_channels, 64)

    def forward(self, x_dict, edge_index_dict,batch,batch_size,sldata):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x.float()).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        # output node representation 
        for node_type in x_dict.keys():
            x_dict[node_type]=self.lin(x_dict[node_type])
        

        node_rep=x_dict['gene/protein']
        node_set=pd.DataFrame(list(batch['gene/protein'].n_id[:batch_size].squeeze().detach().cpu().numpy()))
        node_set.drop_duplicates(inplace=True,keep='first')
        node_set[1]=range(node_set.shape[0])
        node_map=dict(zip(node_set[0],node_set[1]))
    
        prediction_edge=sldata[[0,1]]
        nodea,nodeb=prediction_edge[0],prediction_edge[1]
        nodea=nodea.map(node_map)
        nodeb=nodeb.map(node_map)
        nodea_kgemb=node_rep[nodea.values]
        nodeb_kgemb=node_rep[nodeb.values]

        return nodea_kgemb,nodeb_kgemb
        


        
class CellLineGraphEncoder(torch.nn.Module):
    def __init__(self,data, hidden_channels, num_layers,out_channels):
        super().__init__()

        self.node_encoder = Linear(16, hidden_channels)
        self.edge_encoder = Linear(1, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, 64)

    def forward(self, data,device):
        cellnx_emb=[]
        for current_nx in data:
            
            x=current_nx.x.to(device)
            edge_attr=current_nx.edge_attr.to(device)
            edge_index=current_nx.edge_index.to(device)
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)
            x = self.layers[0].conv(x, edge_index, edge_attr)
            for layer in self.layers[1:]:
                x = layer(x, edge_index, edge_attr)

            x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training)
            average_pooled_emb =F.adaptive_avg_pool2d(x.unsqueeze(0), (1, 32)).squeeze(0)
            max_pooled_emb=F.adaptive_max_pool2d(x.unsqueeze(0), (1,32)).squeeze(0)
            concated_cellemb=torch.cat((average_pooled_emb,max_pooled_emb),dim=1)
            concated_cellemb=self.lin(concated_cellemb)
            cellnx_emb.append(concated_cellemb)


        return cellnx_emb
    








class SequenceEncoder(torch.nn.Module):
    def __init__(self,hidden_channels, out_channels):
        super().__init__()
        self.Linear_seqcell=torch.nn.Sequential(torch.nn.Linear(1280,hidden_channels),torch.nn.Linear(hidden_channels,out_channels))
        self.Linear_seqprotein=torch.nn.Sequential(torch.nn.Linear(1280,hidden_channels),torch.nn.Linear(hidden_channels,out_channels))
    def forward(self,cell_line_protein,protein_emb,device):
        cellseq_emb=[]
        for i in range(3):
            cell_data=cell_line_protein[cell_line_protein['cell_id']==i]
            cell_proteins=set(cell_data['primekg_index'].values)
            cell_proteinsemb=torch.stack([protein_emb[i] for i in cell_proteins])

            average_pooled_emb =F.adaptive_avg_pool2d(cell_proteinsemb.unsqueeze(0), (1, 640)).squeeze(0)
            max_pooled_emb=F.adaptive_max_pool2d(cell_proteinsemb.unsqueeze(0), (1,640)).squeeze(0)
           
            concated_cellemb=torch.cat((average_pooled_emb,max_pooled_emb),dim=1).to(device)
            cell_emb=self.Linear_seqcell(concated_cellemb)
            cellseq_emb.append(cell_emb)

        proteinseq={}
        for k in protein_emb:
            proteinseq[k]=self.Linear_seqprotein(protein_emb[k].to(device))
        return proteinseq,torch.stack(cellseq_emb)
        



# class OmicsEncoder(torch.nn.Module):
#     def __init__(self, out_channels) -> None:
#         super().__init__()
#         self.out_channles=out_channels
#         # self.Linear_cn=torch.nn.Sequential(torch.nn.Linear(25363,2*out_channels),torch.nn.Linear(2*out_channels,out_channels))
#         # self.Linear_ex=torch.nn.Sequential(torch.nn.Linear(19193,2*out_channels),torch.nn.Linear(2*out_channels,out_channels))
#         self.Linear_proteins=torch.nn.Sequential(torch.nn.Linear(128,2*out_channels),torch.nn.Linear(2*out_channels,out_channels))
#         # self.Linear_cell=torch.nn.Sequential(torch.nn.Linear(2*out_channels,out_channels))

#     def forward(self,exdata,device):
#         #---------------ex----------------
#         # exdata=np.array(exdata['ex'],dtype=np.float64)
#         omics_proteinemb={}
#         protein_index=exdata.index
        
#         exdata=torch.tensor(np.array(exdata,dtype=np.float32)).to(device)

        
#        # proteinemb=self.Linear_proteins(exdata)
#         proteinemb=exdata
#         for k in range(exdata.shape[0]):
#             omics_proteinemb[protein_index[k]]= proteinemb[k]
#        # omics_proteinemb.set_index('Unnamed: 0',inplace=True)
       

#         return omics_proteinemb

class OmicsEncoder(torch.nn.Module):
    def __init__(self,hidden_channels):
        super().__init__()
        self.Linear_seqcell=torch.nn.Sequential(torch.nn.Linear(1280,hidden_channels),torch.nn.Linear(hidden_channels,hidden_channels*2))
       
    def forward(self,new_cl,device):
        cellseq_emb=[]
        for i in range(3):
            cell_data=new_cl[new_cl['cell_id']==i]
            cell_tpm=torch.tensor(cell_data['tpm'].values)
            # cell_proteinsemb=torch.stack([protein_emb[i] for i in cell_proteins])

            average_pooled_emb =F.adaptive_avg_pool1d(cell_tpm.unsqueeze(0), 640)
            max_pooled_emb=F.adaptive_max_pool1d(cell_tpm.unsqueeze(0), (640))
           
            concated_cellemb=torch.cat((average_pooled_emb,max_pooled_emb),dim=1).to(device)
            
            cell_emb=self.Linear_seqcell(concated_cellemb.float())
            cellseq_emb.append(cell_emb)

        # proteinseq={}
        # for k in protein_emb:
        #     proteinseq[k]=self.Linear_seqprotein(protein_emb[k].to(device))
        return torch.stack(cellseq_emb).squeeze(dim=1)



















# class OmicsEncoder(torch.nn.Module):
#     def __init__(self, out_channels) -> None:
#         super().__init__()
#         self.out_channles=out_channels
#         self.Linear_cn=torch.nn.Sequential(torch.nn.Linear(25363,2*out_channels),torch.nn.Linear(2*out_channels,out_channels))
#         self.Linear_ex=torch.nn.Sequential(torch.nn.Linear(19193,2*out_channels),torch.nn.Linear(2*out_channels,out_channels))
#         self.Linear_proteins=torch.nn.Sequential(torch.nn.Linear(256,2*out_channels),torch.nn.Linear(2*out_channels,out_channels))
#         self.Linear_cell=torch.nn.Sequential(torch.nn.Linear(2*out_channels,out_channels))

#     def forward(self,exdata,cndata,ex_protein,cn_protein,device):
#         #---------------ex----------------
#         #exdata=np.array(exdata['ex'],dtype=np.float64)
        
#         exdata=torch.tensor(np.array(exdata,dtype=np.float32)).to(device)

        
#         ex_emb=self.Linear_ex(exdata)
#         #---------------cn-----------------
#         #cndata=np.array(cndata['cn'],dtype=np.float64)
#         max_length = max(len(cndata[0][0]), len(cndata[1][0]), len(cndata[2][0]))

#         array1=np.array(cndata[0][0])
#         array2=np.array(cndata[1][0])
#         array3=np.array(cndata[2][0])

#         array1_padded = np.pad(array1, pad_width=( (0, max_length - array1.shape[0])), mode='constant', constant_values=0)
#         array2_padded = np.pad(array2,  pad_width=( (0, max_length - array2.shape[0])), mode='constant', constant_values=0)
#         array3_padded = np.pad(array3,  pad_width=( (0, max_length - array3.shape[0])), mode='constant', constant_values=0)
        
#         cndata=torch.tensor(np.array([array1_padded,array2_padded,array3_padded],dtype=np.float32)).to(device)
        
#         cn_emb=self.Linear_cn(cndata)
#         cell_omicsemb=torch.cat((ex_emb,cn_emb),dim=1).to(device)
#         cell_omicsemb=self.Linear_cell(cell_omicsemb)
#         #-----------protein-----------------------
#         ex_protein=ex_protein.set_index('Unnamed: 0')
#         cn_protein=cn_protein.set_index('Unnamed: 0')
#         exprotein_emb=ex_protein.sort_index()
#         cnprotein_emb=cn_protein.sort_index()
#         protein_index=list(exprotein_emb.index)
#         exprotein_emb=torch.from_numpy(exprotein_emb.to_numpy())
#         cnprotein_emb=torch.from_numpy(cnprotein_emb.to_numpy())
#         omics_proteinemb=torch.cat((exprotein_emb,cnprotein_emb),dim=1)
#         average_pooled_emb =F.adaptive_avg_pool2d(omics_proteinemb.unsqueeze(0), (1300, 128)).squeeze(0)
#         max_pooled_emb=F.adaptive_max_pool2d(omics_proteinemb.unsqueeze(0), (1300,128)).squeeze(0)
#         concated_omics_proteinemb=torch.cat((average_pooled_emb,max_pooled_emb),dim=1).to(device)
#         proteinemb=self.Linear_proteins(concated_omics_proteinemb.to(torch.float32)).to(device)
#         omics_proteinemb={}
#         for k in range(proteinemb.shape[0]):
#             omics_proteinemb[protein_index[k]]=proteinemb[k]

#         return omics_proteinemb,cell_omicsemb


    