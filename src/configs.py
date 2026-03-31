from pathlib import Path

from yacs.config import CfgNode as CN


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(*parts):
    return str(PROJECT_ROOT.joinpath(*parts))

_C=CN()


_C.EXPERIMENT=CN()
_C.EXPERIMENT.NAME='MiT4SL'
_C.EXPERIMENT.SETTING='cross_cell_line'
_C.EXPERIMENT.REPEAT_MODE='seed_repeats'
_C.EXPERIMENT.NUM_RUNS=5
_C.EXPERIMENT.SPLIT_INDEX=0


_C.DATA=CN()
_C.DATA.TPM_THRESHOLD=400


# BKG encoder
_C.KG=CN()
_C.KG.NAME='PrimeKG'
_C.KG.USE_KG=True
_C.KG.NUM_LAYERS=3
_C.KG.EMB_DIM=64
_C.KG.HIDEEN_DIM=64
_C.KG.NUM_HEADS=4

#BKG sampler
_C.KG_SAMPLER=CN()
_C.KG_SAMPLER.SAMPLE_NODES=512
_C.KG_SAMPLER.SAMPLE_LAYERS=8

# Protein sequence encoder
_C.ProteinSeq=CN()
_C.ProteinSeq.NAME='ESM2'
_C.ProteinSeq.USE_Seq=True
_C.ProteinSeq.HIDDEN_DIM=640
_C.ProteinSeq.OUT_DIM=64

# Cell line encoder
_C.Cell_Line=CN()
_C.Cell_Line.USE_Cell=True
_C.Cell_Line.HIDDEN_DIM=64
_C.Cell_Line.EMB_DIM=64
_C.Cell_Line.NUM_LAYERS=64



#MLP  SL classifer
_C.MLP=CN()
_C.MLP.SL_PREDICTOR=CN()
_C.MLP.SL_PREDICTOR.INPUT_DIM=384
_C.MLP.SL_PREDICTOR.HIDDEN_DIM=64
_C.MLP.SL_PREDICTOR.OUT_DIM=2


#MLP DL classifer
_C.MLP.DL_PREDICTOR=CN()
_C.MLP.DL_PREDICTOR.INPUT_DIM=256
_C.MLP.DL_PREDICTOR.HIDDEN_DIM=64
_C.MLP.DL_PREDICTOR.OUT_DIM=2


_C.OPTIM=CN()
_C.OPTIM.NAME='Adam'
_C.OPTIM.LR=1e-3
_C.OPTIM.LR_SEARCH_SPACE=[1e-3, 3e-3]
_C.OPTIM.CROSS_CELL_LINE_SMALL_LR=3e-3
_C.OPTIM.CROSS_CELL_LINE_LARGE_LR=1e-3
_C.OPTIM.BETA1=0.9
_C.OPTIM.BETA2=0.999
_C.OPTIM.EPS=1e-8
_C.OPTIM.WEIGHT_DECAY=0.0


_C.LOSS=CN()
_C.LOSS.LAMBDA1=0.2
_C.LOSS.LAMBDA2=0.2


_C.TRAIN=CN()
_C.TRAIN.BATCH_SIZE=512
_C.TRAIN.BATCH_POS_NEG_RATIO=1
_C.TRAIN.CELL_LINE_SPECIFIC_MAX_EPOCHS=150
_C.TRAIN.CROSS_CELL_LINE_POLICY='by_train_size'
_C.TRAIN.CROSS_CELL_LINE_TRAIN_SIZE_THRESHOLD=5800
_C.TRAIN.CROSS_CELL_LINE_SMALL_MAX_EPOCHS=120
_C.TRAIN.CROSS_CELL_LINE_LARGE_MAX_EPOCHS=20
_C.TRAIN.EARLY_STOPPING=CN()
_C.TRAIN.EARLY_STOPPING.ENABLED=False
_C.TRAIN.EARLY_STOPPING.MONITOR='valid_auc'
_C.TRAIN.EARLY_STOPPING.PATIENCE=5


#SOLVER
_C.SOLVER=CN()
_C.SOLVER.DEVICE=0
_C.SOLVER.USE_DATA='KG_Seq_Cell_Line'
_C.SOLVER.SCENARIO='cross_cell_line'
_C.SOLVER.CELL='Multi_5_to_A549'
_C.SOLVER.NUM_WORKERS=16
_C.SOLVER.KG_DATAPATH=_project_path('data', 'MultiOmics_feature', 'kg_data', 'kgdata.pkl')
_C.SOLVER.KG_NODE_DICT=_project_path('data', 'MultiOmics_feature', 'kg_data', 'node_index_dic.json')
_C.SOLVER.CELLNX_DATAPATH=_project_path('data', 'MultiOmics_feature', 'cell_line_data', 'protein_nx', 'Multi_6_cell_lines_subgraph.pkl')
_C.SOLVER.PROTEINSeq_DATAPATH=_project_path('data', 'MultiOmics_feature', 'seq_data', 'protein_sequence_embedding.pkl')
_C.SOLVER.CELLPROTEIN_DATAPATH=_project_path('data', 'MultiOmics_feature', 'cell_line_data', 'protein_csv', 'Multi_6_cell_lines_proteins.csv')
_C.SOLVER.TASK_DATAPATH=_project_path('data', 'SLbench', 'Scenario', 'Cross_cell_line')
_C.SOLVER.TASK_CELL_TEMPLATE='{cell}'
_C.SOLVER.NODE_TYPE='gene/protein'
_C.SOLVER.REPEAT_EXP_SEED=42
_C.SOLVER.MODEL_SEED=1024
_C.SOLVER.NEGATIVE_SAMPLING_SEED_BASE=42

#Scheduler
_C.SCHEDULER=CN()
_C.SCHEDULER.USE_SCHEDULER=True
_C.SCHEDULER.STEP_SIZE=5
_C.SCHEDULER.GAMMA=0.1


# RESULT 
_C.RESULT=CN()
_C.RESULT.SAVE_MODEL=True
_C.RESULT.SAVE_CHEACKPOINTS_STEP=5
_C.RESULT.LOG_STEPS=1
_C.RESULT.SAVE_PATH=_project_path('result')


def get_cfg_defaults():
    return _C.clone()
