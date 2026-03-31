from pathlib import Path

from configs import get_cfg_defaults


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG_PATHS = [
    './configs/cross_cell_line/protocol.yaml',
    './configs/cross_cell_line/Multi_5_to_A549.yaml',
]
PATH_FIELDS = (
    ('SOLVER', 'KG_DATAPATH'),
    ('SOLVER', 'KG_NODE_DICT'),
    ('SOLVER', 'CELLNX_DATAPATH'),
    ('SOLVER', 'PROTEINSeq_DATAPATH'),
    ('SOLVER', 'CELLPROTEIN_DATAPATH'),
    ('SOLVER', 'TASK_DATAPATH'),
    ('RESULT', 'SAVE_PATH'),
)


def load_cfg_from_paths(cfg_paths=None):
    cfg = get_cfg_defaults()
    cfg_paths = list(cfg_paths or DEFAULT_CFG_PATHS)
    for cfg_path in cfg_paths:
        cfg.merge_from_file(cfg_path)
    normalize_cfg_paths(cfg)
    return cfg, cfg_paths


def normalize_cfg_paths(cfg):
    for section_name, field_name in PATH_FIELDS:
        section = getattr(cfg, section_name)
        current_value = getattr(section, field_name)
        if isinstance(current_value, str) and '{cell}' in current_value:
            current_value = current_value.format(cell=cfg.SOLVER.CELL)
        if current_value and not Path(current_value).is_absolute():
            current_value = str((PROJECT_ROOT / current_value).resolve())
        setattr(section, field_name, current_value)


def resolve_task_cell_target(cfg):
    try:
        return cfg.SOLVER.TASK_CELL_TEMPLATE.format(cell=cfg.SOLVER.CELL)
    except KeyError as exc:
        raise KeyError(
            f"Invalid SOLVER.TASK_CELL_TEMPLATE '{cfg.SOLVER.TASK_CELL_TEMPLATE}'. "
            "Only '{cell}' is currently supported."
        ) from exc
