import argparse
import json
import logging
import os
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from info_nce import InfoNCE
from torch.optim import lr_scheduler
from tqdm import tqdm

from config_loader import load_cfg_from_paths, resolve_task_cell_target
from model import MiT4SL
from util import (
    Construct_loader,
    Downstream_data_preprocess_cell,
    compute_split_metrics,
    init_graph_data,
    log_metrics,
    overlapping_with_sequence,
    save_model,
    set_logger,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description='MiT4SL for cell line SL prediction')
    parser.add_argument(
        '--cfg',
        dest='cfg_paths',
        action='append',
        help='Config file path. Pass multiple times to merge shared protocol config(s) and a target config.',
    )
    parser.add_argument(
        '--cfg_path',
        dest='cfg_paths',
        action='append',
        help='Legacy alias for --cfg. Pass multiple times to merge several config files.',
    )
    parser.add_argument(
        '--device',
        help="Optional runtime device override. Accepts a GPU index (for example, 0), 'cpu', 'cuda', or 'cuda:<index>'.",
        type=str,
        default=None,
    )
    parser.add_argument('--Save_model_path', help='Optional output directory override', type=str, default=None)
    return parser.parse_args()



def load_cfg(args):
    return load_cfg_from_paths(args.cfg_paths)


def apply_runtime_overrides(cfg, args):
    if args.device is None:
        return

    device_override = str(args.device).strip()
    if not device_override:
        raise ValueError('--device override must not be empty.')

    cfg.SOLVER.DEVICE = int(device_override) if device_override.isdigit() else device_override


def resolve_runtime_device(device_setting):
    if isinstance(device_setting, str):
        normalized = device_setting.strip().lower()
        if not normalized:
            raise ValueError('SOLVER.DEVICE must not be empty.')
        if normalized == 'cpu':
            return torch.device('cpu')
        if normalized == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError(f"Requested device '{device_setting}', but CUDA is not available.")
            return torch.device('cuda')
        if normalized.startswith('cuda:'):
            if not torch.cuda.is_available():
                raise ValueError(f"Requested device '{device_setting}', but CUDA is not available.")
            device_suffix = normalized.split(':', 1)[1]
            if not device_suffix.isdigit():
                raise ValueError(
                    "Unsupported device override. Use an integer GPU index, 'cpu', 'cuda', or 'cuda:<index>'."
                )
            device_setting = int(device_suffix)
        if normalized.isdigit():
            device_setting = int(normalized)
        else:
            raise ValueError(
                "Unsupported device override. Use an integer GPU index, 'cpu', 'cuda', or 'cuda:<index>'."
            )

    if torch.cuda.is_available():
        device_index = int(device_setting)
        visible_device_count = torch.cuda.device_count()
        if device_index < 0 or device_index >= visible_device_count:
            raise ValueError(
                f"Requested cuda:{device_index}, but only {visible_device_count} CUDA device(s) are visible."
            )
        return torch.device(f"cuda:{device_index}")

    return torch.device('cpu')



def _format_token(value):
    return str(value).replace('/', '-').replace('.', 'p')



def build_output_dir(cfg, override_path=None, effective_lr=None):
    if override_path:
        return override_path
    del effective_lr
    return os.path.join(cfg.RESULT.SAVE_PATH, cfg.EXPERIMENT.SETTING, cfg.SOLVER.CELL)


def build_cross_cell_line_presets(cfg):
    threshold = int(cfg.TRAIN.CROSS_CELL_LINE_TRAIN_SIZE_THRESHOLD)
    return {
        'large': {
            'source_train_size_min': threshold,
            'effective_lr': cfg.OPTIM.CROSS_CELL_LINE_LARGE_LR,
            'effective_max_epochs': cfg.TRAIN.CROSS_CELL_LINE_LARGE_MAX_EPOCHS,
        },
        'small': {
            'source_train_size_max': threshold - 1,
            'effective_lr': cfg.OPTIM.CROSS_CELL_LINE_SMALL_LR,
            'effective_max_epochs': cfg.TRAIN.CROSS_CELL_LINE_SMALL_MAX_EPOCHS,
        },
    }


def is_recommendation_scenario(cfg):
    return 'recom' in cfg.SOLVER.SCENARIO.lower()


def format_metric_label(metric_name):
    if metric_name.startswith('ndcg_'):
        return f"NDCG@{metric_name.split('_', 1)[1]}"
    if metric_name.startswith('precision_'):
        return f"Precision@{metric_name.split('_', 1)[1]}"
    return metric_name.upper()


def save_resolved_config(cfg, output_dir, cfg_paths, source_train_size, effective_lr, effective_max_epochs, size_bucket):
    os.makedirs(output_dir, exist_ok=True)
    resolved_cfg = cfg.clone()
    resolved_cfg.OPTIM.LR = effective_lr
    with open(os.path.join(output_dir, 'resolved_config.yaml'), 'w') as f:
        f.write(resolved_cfg.dump())

    metadata = {
        'config_files': cfg_paths,
        'experiment_setting': cfg.EXPERIMENT.SETTING,
        'repeat_mode': cfg.EXPERIMENT.REPEAT_MODE,
        'num_runs': cfg.EXPERIMENT.NUM_RUNS,
        'split_index': cfg.EXPERIMENT.SPLIT_INDEX,
        'source_train_size': int(source_train_size),
        'size_bucket': size_bucket,
        'task_cell_template': cfg.SOLVER.TASK_CELL_TEMPLATE,
        'resolved_task_cell': resolve_task_cell_target(cfg),
        'cross_cell_line_policy': cfg.TRAIN.CROSS_CELL_LINE_POLICY,
        'cross_cell_line_train_size_threshold': int(cfg.TRAIN.CROSS_CELL_LINE_TRAIN_SIZE_THRESHOLD),
        'effective_max_epochs': int(effective_max_epochs),
        'lr_search_space': list(cfg.OPTIM.LR_SEARCH_SPACE),
        'lr_presets': list(cfg.OPTIM.LR_SEARCH_SPACE),
        'selected_lr': effective_lr,
        'batch_size': cfg.TRAIN.BATCH_SIZE,
        'device': cfg.SOLVER.DEVICE,
        'tpm_threshold': cfg.DATA.TPM_THRESHOLD,
    }
    if is_recommendation_scenario(cfg):
        metadata['recommendation_topk'] = [5, 10, 20]
    if cfg.EXPERIMENT.SETTING == 'cross_cell_line':
        metadata['cross_cell_line_presets'] = build_cross_cell_line_presets(cfg)
    with open(os.path.join(output_dir, 'run_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)



def resolve_cross_cell_line_bucket(cfg, source_train_size):
    if cfg.TRAIN.CROSS_CELL_LINE_POLICY != 'by_train_size':
        raise ValueError(f"Unsupported cross-cell-line hyperparameter policy: {cfg.TRAIN.CROSS_CELL_LINE_POLICY}")
    if source_train_size >= cfg.TRAIN.CROSS_CELL_LINE_TRAIN_SIZE_THRESHOLD:
        return 'large'
    return 'small'



def resolve_effective_hyperparameters(cfg, source_train_size):
    if cfg.EXPERIMENT.SETTING == 'cell_line_specific':
        return {
            'size_bucket': 'fixed',
            'effective_lr': cfg.OPTIM.LR,
            'effective_max_epochs': cfg.TRAIN.CELL_LINE_SPECIFIC_MAX_EPOCHS,
        }

    if cfg.EXPERIMENT.SETTING == 'cross_cell_line':
        size_bucket = resolve_cross_cell_line_bucket(cfg, source_train_size)
        if size_bucket == 'large':
            return {
                'size_bucket': size_bucket,
                'effective_lr': cfg.OPTIM.CROSS_CELL_LINE_LARGE_LR,
                'effective_max_epochs': cfg.TRAIN.CROSS_CELL_LINE_LARGE_MAX_EPOCHS,
            }
        return {
            'size_bucket': size_bucket,
            'effective_lr': cfg.OPTIM.CROSS_CELL_LINE_SMALL_LR,
            'effective_max_epochs': cfg.TRAIN.CROSS_CELL_LINE_SMALL_MAX_EPOCHS,
        }

    raise ValueError(f"Unsupported experiment setting: {cfg.EXPERIMENT.SETTING}")


def resolve_run_context(cfg, run_idx):
    if cfg.EXPERIMENT.REPEAT_MODE == 'seed_repeats':
        return cfg.EXPERIMENT.SPLIT_INDEX, cfg.SOLVER.MODEL_SEED + run_idx
    if cfg.EXPERIMENT.REPEAT_MODE == 'split_repeats':
        return run_idx, cfg.SOLVER.MODEL_SEED
    raise ValueError(f"Unsupported repeat mode: {cfg.EXPERIMENT.REPEAT_MODE}")



def inspect_reference_source_train_size(cfg, node_type_dict):
    split_index, _ = resolve_run_context(cfg, 0)
    task_cell_target = resolve_task_cell_target(cfg)
    train_data, val_data, *_ = Downstream_data_preprocess_cell(
        cfg.SOLVER.TASK_DATAPATH,
        task_cell_target,
        node_type_dict,
        split_index,
    )
    return split_index, len(train_data) + len(val_data)



def resolve_negative_sampling_seed(cfg, epoch, sample_seed):
    if int(cfg.SOLVER.NEGATIVE_SAMPLING_SEED_BASE) >= 0:
        return epoch + int(cfg.SOLVER.NEGATIVE_SAMPLING_SEED_BASE)
    return epoch + sample_seed + cfg.SOLVER.REPEAT_EXP_SEED



def build_balanced_training_pairs(cfg, epoch, sample_seed, sldata, ori_train_data):
    batch_sl = sldata[sldata[3] == 1].reset_index(drop=True)
    batch_nosl = sldata[sldata[3] == 0].reset_index(drop=True)
    ori_batch_sl = ori_train_data[ori_train_data[3] == 1].reset_index(drop=True)
    ori_batch_nosl = ori_train_data[ori_train_data[3] == 0].reset_index(drop=True)

    if batch_sl.empty or batch_nosl.empty:
        raise ValueError('Training data must contain both positive and negative SL pairs.')

    rng = np.random.default_rng(resolve_negative_sampling_seed(cfg, epoch, sample_seed))
    sampled_idx = rng.choice(
        batch_nosl.index.to_numpy(),
        size=batch_sl.shape[0] * cfg.TRAIN.BATCH_POS_NEG_RATIO,
        replace=True,
    )

    sampled_sldata = pd.concat([batch_sl, batch_nosl.iloc[sampled_idx]], axis=0).reset_index(drop=True)
    sampled_ori_data = pd.concat([ori_batch_sl, ori_batch_nosl.iloc[sampled_idx]], axis=0).reset_index(drop=True)

    permutation = rng.permutation(sampled_sldata.shape[0])
    sampled_sldata = sampled_sldata.iloc[permutation].reset_index(drop=True)
    sampled_ori_data = sampled_ori_data.iloc[permutation].reset_index(drop=True)
    return sampled_sldata, sampled_ori_data



def train(cfg, epoch, sample_seed, context_mit4sl, train_loader, optimizer_model, sldata, num_train_node, ori_train_data, device):
    criterion = nn.CrossEntropyLoss()
    context_mit4sl.train()
    infoloss = InfoNCE()
    mseloss = nn.MSELoss()
    loss_values = []
    scheduler_model = maybe_build_scheduler(cfg, optimizer_model)

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        batch = batch.to(device)
        balanced_sldata, balanced_ori_train_data = build_balanced_training_pairs(cfg, epoch, sample_seed, sldata, ori_train_data)
        labels = torch.tensor(balanced_sldata[3].values).to(device)
        _, tri_emb1, tri_emb2, prediction_result, average_prediction = context_mit4sl(
            balanced_sldata,
            batch,
            num_train_node,
            balanced_ori_train_data,
        )
        loss_cl = infoloss(tri_emb1, tri_emb2)
        predicted_labels = torch.argmax(average_prediction, dim=1)
        loss_mse = mseloss(predicted_labels.float(), labels.float())
        loss_cls = criterion(prediction_result, labels.long())
        loss = loss_cls + cfg.LOSS.LAMBDA1 * loss_cl + cfg.LOSS.LAMBDA2 * loss_mse

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        if scheduler_model is not None:
            scheduler_model.step()
        loss_values.append(float(loss.detach().cpu().item()))

    return {'loss': float(np.mean(loss_values)) if loss_values else 0.0}



def evaluate_split(
    context_mit4sl,
    data_loader,
    pair_data,
    num_nodes,
    ori_data,
    device,
    include_classification_metrics=True,
    include_recommendation_metrics=False,
):
    all_prediction_label, all_prediction_result = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            prediction_label = pair_data[3]
            _, _, _, prediction_result, _ = context_mit4sl(pair_data, batch, num_nodes, ori_data)
            all_prediction_label.append(torch.tensor(prediction_label.values).to(device))
            all_prediction_result.append(prediction_result)

    all_prediction_label = torch.cat(all_prediction_label)
    all_prediction_result = torch.cat(all_prediction_result)
    return compute_split_metrics(
        all_prediction_label,
        all_prediction_result,
        include_classification_metrics=include_classification_metrics,
        include_recommendation_metrics=include_recommendation_metrics,
    )



def evaluate(context_mit4sl, val_loader, test_loader, valdata, testdata, num_val_node, num_test_node, ori_val_data, ori_test_data, device, include_recommendation_metrics=False):
    context_mit4sl.eval()
    valid_metrics = evaluate_split(context_mit4sl, val_loader, valdata, num_val_node, ori_val_data, device)
    test_metrics = evaluate_split(
        context_mit4sl,
        test_loader,
        testdata,
        num_test_node,
        ori_test_data,
        device,
        include_classification_metrics=not include_recommendation_metrics,
        include_recommendation_metrics=include_recommendation_metrics,
    )
    valid_log = {f'valid_{metric}': value for metric, value in valid_metrics.items()}
    return valid_log, test_metrics



def maybe_build_scheduler(cfg, optimizer_model):
    if not cfg.SCHEDULER.USE_SCHEDULER:
        return None
    return lr_scheduler.StepLR(
        optimizer_model,
        step_size=cfg.SCHEDULER.STEP_SIZE,
        gamma=cfg.SCHEDULER.GAMMA,
    )



def should_early_stop(cfg):
    return cfg.TRAIN.EARLY_STOPPING.ENABLED



def resolve_monitor_value(cfg, valid_log):
    monitor_metric = cfg.TRAIN.EARLY_STOPPING.MONITOR
    if monitor_metric not in valid_log:
        raise KeyError(f"Monitor metric '{monitor_metric}' not found in validation metrics: {sorted(valid_log.keys())}")
    return monitor_metric, valid_log[monitor_metric]



def main():
    args = parse_args()
    cfg, cfg_paths = load_cfg(args)
    apply_runtime_overrides(cfg, args)

    # Match the legacy MiT4SL_RECOMB global RNG initialization before preprocessing.
    torch.manual_seed(0)
    np.random.seed(0)

    if cfg.SOLVER.TASK_DATAPATH is None:
        raise ValueError('TASK_DATAPATH must be provided.')
    if cfg.SOLVER.USE_DATA != 'KG_Seq_Cell_Line':
        raise ValueError('Only KG_Seq_Cell_Line mode is currently supported.')

    with open(cfg.SOLVER.KG_NODE_DICT, 'r') as f:
        node_index = json.load(f)

    gene_protein = node_index[cfg.SOLVER.NODE_TYPE]
    node_type = cfg.SOLVER.NODE_TYPE
    task_cell_target = resolve_task_cell_target(cfg)
    reference_split_index, reference_source_train_size = inspect_reference_source_train_size(cfg, gene_protein)
    reference_hparams = resolve_effective_hyperparameters(cfg, reference_source_train_size)

    args.Save_model_path = build_output_dir(cfg, args.Save_model_path, reference_hparams['effective_lr'])
    os.makedirs(args.Save_model_path, exist_ok=True)

    device = resolve_runtime_device(cfg.SOLVER.DEVICE)
    set_logger(args)

    logging.info('Model: %s', cfg.EXPERIMENT.NAME)
    logging.info('Config files: %s', ', '.join(cfg_paths))
    logging.info('Experiment setting: %s', cfg.EXPERIMENT.SETTING)
    logging.info('Repeat mode: %s (%d runs)', cfg.EXPERIMENT.REPEAT_MODE, cfg.EXPERIMENT.NUM_RUNS)
    logging.info('LR presets: %s', cfg.OPTIM.LR_SEARCH_SPACE)
    logging.info('Selected learning rate (reference split): %s', reference_hparams['effective_lr'])
    logging.info('Selection monitor: %s', cfg.TRAIN.EARLY_STOPPING.MONITOR)
    logging.info('Batch size: %d', cfg.TRAIN.BATCH_SIZE)
    logging.info('TPM threshold: %s', cfg.DATA.TPM_THRESHOLD)
    logging.info('Device setting: %s', cfg.SOLVER.DEVICE)
    logging.info('Resolved runtime device: %s', device)
    logging.info('Task cell template: %s', cfg.SOLVER.TASK_CELL_TEMPLATE)
    logging.info('Resolved task cell: %s', task_cell_target)
    if cfg.EXPERIMENT.SETTING == 'cross_cell_line':
        logging.info('Cross-cell-line policy: %s', cfg.TRAIN.CROSS_CELL_LINE_POLICY)
        logging.info('Cross-cell-line train size threshold: %d', cfg.TRAIN.CROSS_CELL_LINE_TRAIN_SIZE_THRESHOLD)
        cross_cell_line_presets = build_cross_cell_line_presets(cfg)
        for bucket_name, preset in cross_cell_line_presets.items():
            size_range = (
                f">= {preset['source_train_size_min']}"
                if 'source_train_size_min' in preset
                else f"<= {preset['source_train_size_max']}"
            )
            logging.info(
                'Cross-cell-line %s preset (source_train_size %s): lr=%s, max_epochs=%d',
                bucket_name,
                size_range,
                preset['effective_lr'],
                preset['effective_max_epochs'],
            )
    logging.info('Reference split_index = %d', reference_split_index)
    logging.info('Reference source_train_size = %d', reference_source_train_size)
    logging.info('Reference size_bucket = %s', reference_hparams['size_bucket'])
    logging.info('Reference effective_max_epochs = %d', reference_hparams['effective_max_epochs'])
    kgdata, cell_ppidata = init_graph_data(cfg.SOLVER.KG_DATAPATH, cfg.SOLVER.CELLNX_DATAPATH)
    new_proteinseq_data, cell_line_proteins = overlapping_with_sequence(
        cfg.SOLVER.PROTEINSeq_DATAPATH,
        cfg.SOLVER.CELLPROTEIN_DATAPATH,
        cfg.SOLVER.TASK_DATAPATH,
        task_cell_target,
        cfg.EXPERIMENT.SPLIT_INDEX,
    )

    logging.info('KG data: %s', cfg.KG.NAME)
    logging.info('Seq data: %s', cfg.ProteinSeq.NAME)
    logging.info('Cell line target: %s', cfg.SOLVER.CELL)
    logging.info('Data used: %s', cfg.SOLVER.USE_DATA)

    result_text = f"""
                           {cfg.EXPERIMENT.NAME}
                        {cfg.SOLVER.CELL}
                    ----------------------------
    """
    eval_metric_runs = {'run': [], 'selected_epoch': []}
    summary_metric_names = None

    for run_idx in range(cfg.EXPERIMENT.NUM_RUNS):
        split_index, run_seed = resolve_run_context(cfg, run_idx)
        logging.info('Run_%d training... (split_index=%d, seed=%d)', run_idx, split_index, run_seed)
        set_seed(run_seed)

        train_data, val_data, test_data, train_mask, val_mask, test_mask, num_train_node, num_val_node, num_test_node, ori_train_data, ori_val_data, ori_test_data = Downstream_data_preprocess_cell(
            cfg.SOLVER.TASK_DATAPATH,
            task_cell_target,
            gene_protein,
            split_index,
        )
        source_train_size = len(train_data) + len(val_data)
        effective_hparams = resolve_effective_hyperparameters(cfg, source_train_size)
        effective_lr = effective_hparams['effective_lr']
        effective_max_epochs = effective_hparams['effective_max_epochs']
        size_bucket = effective_hparams['size_bucket']

        context_mit4sl = MiT4SL(
            kgdata,
            cell_ppidata,
            new_proteinseq_data,
            cell_line_proteins,
            cfg.KG.HIDEEN_DIM,
            cfg.KG.EMB_DIM,
            cfg.KG.NUM_HEADS,
            cfg.KG.NUM_LAYERS,
            cfg.Cell_Line.HIDDEN_DIM,
            cfg.Cell_Line.NUM_LAYERS,
            cfg.KG.USE_KG,
            cfg.Cell_Line.USE_Cell,
            cfg.ProteinSeq.USE_Seq,
            device,
        )

        if cfg.OPTIM.NAME.lower() != 'adam':
            raise ValueError(f"Unsupported optimizer: {cfg.OPTIM.NAME}")

        optimizer_model = optim.Adam(
            context_mit4sl.parameters(),
            lr=effective_lr,
            betas=(cfg.OPTIM.BETA1, cfg.OPTIM.BETA2),
            eps=cfg.OPTIM.EPS,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        )
        if run_idx == 0:
            save_resolved_config(
                cfg,
                args.Save_model_path,
                cfg_paths,
                source_train_size,
                effective_lr,
                effective_max_epochs,
                size_bucket,
            )

        logging.info('source_train_size = %d', source_train_size)
        logging.info('size_bucket = %s', size_bucket)
        logging.info('effective_learning_rate = %s', effective_lr)
        logging.info('effective_max_epochs = %d', effective_max_epochs)
        logging.info('num_train_node = %d', num_train_node)
        logging.info('num_test_node = %d', num_test_node)

        train_loader, val_loader, test_loader = Construct_loader(
            cfg.KG_SAMPLER.SAMPLE_NODES,
            cfg.KG_SAMPLER.SAMPLE_LAYERS,
            cfg.SOLVER.NUM_WORKERS,
            kgdata,
            train_mask,
            val_mask,
            test_mask,
            node_type,
            num_train_node,
            num_val_node,
            num_test_node,
        )

        recent_training_logs, recent_val_logs, recent_testing_logs = [], [], []
        best_monitor = float('-inf')
        best_test_metrics = None
        last_test_metrics = None
        selected_epoch = 0
        patience_counter = 0

        for epoch in range(1, effective_max_epochs + 1):
            train_log = train(
                cfg,
                epoch,
                run_seed,
                context_mit4sl,
                train_loader,
                optimizer_model,
                train_data,
                num_train_node,
                ori_train_data,
                device,
            )
            valid_log, test_metrics = evaluate(
                context_mit4sl,
                val_loader,
                test_loader,
                val_data,
                test_data,
                num_val_node,
                num_test_node,
                ori_val_data,
                ori_test_data,
                device,
                include_recommendation_metrics=is_recommendation_scenario(cfg),
            )

            recent_training_logs.append(train_log)
            recent_val_logs.append(valid_log)
            recent_testing_logs.append(test_metrics)
            last_test_metrics = test_metrics.copy()
            selected_epoch = epoch

            monitor_metric, current_monitor = resolve_monitor_value(cfg, valid_log)
            if current_monitor > best_monitor:
                best_monitor = current_monitor
                best_test_metrics = test_metrics.copy()
                patience_counter = 0
            else:
                if should_early_stop(cfg):
                    patience_counter += 1

            if epoch % cfg.RESULT.LOG_STEPS == 0:
                training_metrics = {
                    metric: sum(log[metric] for log in recent_training_logs) / len(recent_training_logs)
                    for metric in recent_training_logs[0]
                }
                val_metrics = {
                    metric: sum(log[metric] for log in recent_val_logs) / len(recent_val_logs)
                    for metric in recent_val_logs[0]
                }
                testing_metrics = {
                    metric: sum(log[metric] for log in recent_testing_logs) / len(recent_testing_logs)
                    for metric in recent_testing_logs[0]
                }
                logging.info('============= Start Training ... ==============')
                log_metrics('Training average', epoch, training_metrics)
                logging.info('============= Start Validating ... ============')
                log_metrics('Valid average', epoch, val_metrics)
                logging.info('============= Start Testing ... ===============')
                log_metrics('Testing average', epoch, testing_metrics)
                recent_training_logs, recent_val_logs, recent_testing_logs = [], [], []

            if cfg.RESULT.SAVE_MODEL and epoch % cfg.RESULT.SAVE_CHEACKPOINTS_STEP == 0:
                save_model(context_mit4sl, optimizer_model, args.Save_model_path)

            if should_early_stop(cfg) and patience_counter >= cfg.TRAIN.EARLY_STOPPING.PATIENCE:
                logging.info('Early stopped at epoch %d based on %s.', epoch, cfg.TRAIN.EARLY_STOPPING.MONITOR)
                break

        if cfg.RESULT.SAVE_MODEL:
            save_model(context_mit4sl, optimizer_model, args.Save_model_path)

        report_test_metrics = last_test_metrics or best_test_metrics
        if summary_metric_names is None:
            summary_metric_names = list(report_test_metrics.keys())
            for metric_name in summary_metric_names:
                eval_metric_runs[metric_name] = []

        for metric_name in summary_metric_names:
            print(f"run_{run_idx}_{metric_name}:{round(report_test_metrics[metric_name], 4)}")

        eval_metric_runs['run'].append(run_idx)
        eval_metric_runs['selected_epoch'].append(selected_epoch)
        for metric_name in summary_metric_names:
            eval_metric_runs[metric_name].append(round(report_test_metrics[metric_name], 4))

        metric_lines = '\n'.join(
            f"                        {format_metric_label(metric_name)}:{round(report_test_metrics[metric_name], 4)}"
            for metric_name in summary_metric_names
        )
        result_text += f"""

                                run_{run_idx}
                        ----------------------------
                                {metric_lines}
                        ----------------------------
                    """

    eval_metric_runs = pd.DataFrame(eval_metric_runs)
    metric_end = cfg.EXPERIMENT.NUM_RUNS - 1
    eval_metric_runs.loc[cfg.EXPERIMENT.NUM_RUNS, 'run'] = 'average'
    eval_metric_runs.loc[cfg.EXPERIMENT.NUM_RUNS, 'selected_epoch'] = '-'
    eval_metric_runs.loc[cfg.EXPERIMENT.NUM_RUNS + 1, 'run'] = 'std'
    eval_metric_runs.loc[cfg.EXPERIMENT.NUM_RUNS + 1, 'selected_epoch'] = '-'
    aggregate_lines = []
    for metric_name in summary_metric_names:
        mean_value = round(eval_metric_runs.loc[:metric_end, metric_name].mean(), 4)
        std_value = round(eval_metric_runs.loc[:metric_end, metric_name].std(), 4)
        eval_metric_runs.loc[cfg.EXPERIMENT.NUM_RUNS, metric_name] = mean_value
        eval_metric_runs.loc[cfg.EXPERIMENT.NUM_RUNS + 1, metric_name] = std_value
        aggregate_lines.append(f"{format_metric_label(metric_name)}_mean(std):{mean_value}({std_value})")

    result_text += f"""
                        ----------------------------
                    {chr(10).join(aggregate_lines)}
                        ----------------------------
                        """
    with open(os.path.join(args.Save_model_path, f"{cfg.SOLVER.CELL}_results.txt"), 'w') as f:
        f.write(result_text)
    eval_metric_runs.to_csv(os.path.join(args.Save_model_path, 'final_result_eval.csv'), index=False)


if __name__ == "__main__":
    s = time()
    main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
