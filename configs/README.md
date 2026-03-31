# `configs/`

This directory contains the YAML configuration catalog used to run MiT4SL experiments.

A runnable experiment is defined by **two YAML files loaded in order**:

1. `protocol.yaml` — shared settings for one scenario or study.
2. A target-specific YAML file — a lightweight override for a cell line, target dataset, or case-study item.


## What each group is for

| Path | Purpose | Notes |
| --- | --- | --- |
| `cross_cell_line/` | Cross-cell-line transfer/generalization experiments | Uses `EXPERIMENT.SETTING: cross_cell_line`; target files are named `Multi_5_to_*`. |
| `cell_line_specific/random/` | Cell-line-specific benchmark with random splitting | Base scenario under `data/SLbench/Scenario/Cell_line_specific/Random_splitting`. |
| `cell_line_specific/cold_start/` | Cell-line-specific cold-start benchmark | Uses the `Cold_start` data split. |
| `cell_line_specific/long_tail/` | Cell-line-specific long-tail benchmark | Uses the `Tail_node` split. |
| `cell_line_specific/cross_functional/` | Cell-line-specific cross-functional benchmark | Uses the `Cross_functional` split. |
| `recom_sl_partner/` | Synthetic-lethal partner recommendation | Example targets include `A549_KRAS` and `A549_TP53`. |
| `recom_sl_cell_line/dede/` | Cell-line recommendation configs for the Dede dataset/study | Saves outputs under `result/recom_cell_line/Dede`. |
| `recom_sl_cell_line/ito/` | Cell-line recommendation configs for the Ito dataset/study | Saves outputs under `result/recom_cell_line/Ito`. |
| `recom_sl_cell_line/case_study_te1/` | TE-1 case-study configuration | Uses data under `data/Case_study_TE1`. |

## File conventions

- Every **leaf config directory** contains exactly one `protocol.yaml`.
- All other `*.yaml` files beside it are selectable targets.
- `protocol.yaml` usually stores shared sections such as `EXPERIMENT`, `OPTIM`, `LOSS`, `TRAIN`, `SOLVER`, and sometimes `RESULT`.
- Target YAMLs are intentionally small. Most only set `SOLVER.CELL`, while a few also override cell-specific paths or training hyperparameters.
- Relative paths and `{cell}` placeholders are normalized by `src/config_loader.py` before training starts.

The standard launcher is:

```bash
bash scripts/run_mit4sl.sh --config-dir <group> --target <name>
```

Internally, the launcher resolves a matching target YAML in `configs/`, then runs:

```bash
python src/train_MiT4SL.py --cfg <protocol.yaml> --cfg <target.yaml>
```

`src/config_loader.py` merges the YAML files on top of the code defaults in `src/configs.py`, expands `{cell}` placeholders, and converts relative paths such as `./data/...` and `./result/...` into absolute repository paths.


## Directory layout

```text
configs/
├── cross_cell_line/
│   ├── protocol.yaml
│   └── Multi_5_to_{22Rv1,A375,A549,Jurkat,MeWo,Pk1}.yaml
├── recom_sl_partner/
│   ├── protocol.yaml
│   └── A549_{KRAS,TP53}.yaml
├── recom_sl_cell_line/
│   ├── case_study_te1/
│   │   ├── protocol.yaml
│   │   └── TE-1.yaml
│   ├── dede/
│   │   ├── protocol.yaml
│   │   └── {A549,HT29,OVCAR8}.yaml
│   └── ito/
│       ├── protocol.yaml
│       └── {A549,HS936T,HS944T,Meljuso}.yaml
└── cell_line_specific/
    ├── cold_start/
    │   ├── protocol.yaml
    │   └── {22Rv1,A549,Jurkat,MeWo,Pk1}.yaml
    ├── cross_functional/
    │   ├── protocol.yaml
    │   └── {A549,MeWo,Pk1}.yaml
    ├── long_tail/
    │   ├── protocol.yaml
    │   └── {A549,Pk1}.yaml
    └── random/
        ├── protocol.yaml
        └── {22Rv1,A375,A549,Jurkat,MeWo,Pk1}.yaml
```
