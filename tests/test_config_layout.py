import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_loader import PROJECT_ROOT, load_cfg_from_paths, resolve_task_cell_target


class ConfigLoaderBehaviorTest(unittest.TestCase):
    def test_load_cfg_from_paths_resolves_relative_paths_and_cell_placeholders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            protocol = tmpdir / 'protocol.yaml'
            target = tmpdir / 'target.yaml'

            protocol.write_text(
                "\n".join(
                    [
                        "SOLVER:",
                        "  TASK_DATAPATH: './data/example/tasks'",
                        "  TASK_CELL_TEMPLATE: '{cell}/nested'",
                        "  CELLNX_DATAPATH: './data/example/graphs/{cell}.pkl'",
                        "  CELLPROTEIN_DATAPATH: './data/example/proteins/{cell}.csv'",
                        "RESULT:",
                        "  SAVE_PATH: './result/example'",
                    ]
                )
                + "\n"
            )
            target.write_text("SOLVER:\n  CELL: 'A549'\n")

            cfg, cfg_paths = load_cfg_from_paths([str(protocol), str(target)])

            self.assertEqual(cfg_paths, [str(protocol), str(target)])
            self.assertEqual(resolve_task_cell_target(cfg), 'A549/nested')
            self.assertEqual(
                cfg.SOLVER.TASK_DATAPATH,
                str((PROJECT_ROOT / 'data/example/tasks').resolve()),
            )
            self.assertEqual(
                cfg.SOLVER.CELLNX_DATAPATH,
                str((PROJECT_ROOT / 'data/example/graphs/A549.pkl').resolve()),
            )
            self.assertEqual(
                cfg.SOLVER.CELLPROTEIN_DATAPATH,
                str((PROJECT_ROOT / 'data/example/proteins/A549.csv').resolve()),
            )
            self.assertEqual(
                cfg.RESULT.SAVE_PATH,
                str((PROJECT_ROOT / 'result/example').resolve()),
            )


class ConfigCatalogValidationTest(unittest.TestCase):
    def test_all_structured_configs_point_to_existing_repo_data(self):
        protocol_files = sorted((ROOT / 'configs').rglob('protocol.yaml'))
        self.assertTrue(protocol_files, 'No protocol.yaml files found under configs/.')

        for protocol in protocol_files:
            with self.subTest(protocol=str(protocol)):
                target_files = sorted(
                    path
                    for path in protocol.parent.glob('*.yaml')
                    if path.name != 'protocol.yaml'
                )
                self.assertTrue(target_files, f'No target config found beside {protocol}.')

                for target in target_files:
                    with self.subTest(target=str(target)):
                        cfg, _ = load_cfg_from_paths([str(protocol), str(target)])

                        existing_paths = {
                            'KG_DATAPATH': cfg.SOLVER.KG_DATAPATH,
                            'KG_NODE_DICT': cfg.SOLVER.KG_NODE_DICT,
                            'PROTEINSeq_DATAPATH': cfg.SOLVER.PROTEINSeq_DATAPATH,
                            'CELLNX_DATAPATH': cfg.SOLVER.CELLNX_DATAPATH,
                            'CELLPROTEIN_DATAPATH': cfg.SOLVER.CELLPROTEIN_DATAPATH,
                            'TASK_DATAPATH': cfg.SOLVER.TASK_DATAPATH,
                        }
                        for label, raw_path in existing_paths.items():
                            self.assertTrue(
                                Path(raw_path).exists(),
                                f'{protocol} + {target} -> {label} missing: {raw_path}',
                            )

                        self.assertTrue(
                            Path(cfg.RESULT.SAVE_PATH).is_absolute(),
                            f'{protocol} + {target} -> RESULT.SAVE_PATH was not normalized.',
                        )

                        task_dir = Path(cfg.SOLVER.TASK_DATAPATH) / resolve_task_cell_target(cfg)
                        self.assertTrue(
                            task_dir.exists(),
                            f'{protocol} + {target} -> task dir missing: {task_dir}',
                        )
                        self.assertTrue(
                            any(task_dir.glob('sl_train_*.csv')),
                            f'{protocol} + {target} -> no sl_train_*.csv under {task_dir}',
                        )
                        self.assertTrue(
                            any(task_dir.glob('sl_test_*.csv')),
                            f'{protocol} + {target} -> no sl_test_*.csv under {task_dir}',
                        )


if __name__ == '__main__':
    unittest.main()
