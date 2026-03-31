import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'scripts' / 'run_mit4sl.sh'


class RunMiT4SLShellScriptTest(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run(
            ['bash', str(SCRIPT_PATH), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_default_dry_run_targets_a549_cross_cell_line(self):
        completed = self.run_script('--dry-run')

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('configs/cross_cell_line/protocol.yaml', completed.stdout)
        self.assertIn('configs/cross_cell_line/Multi_5_to_A549.yaml', completed.stdout)
        self.assertIn('src/train_MiT4SL.py', completed.stdout)

    def test_list_targets_includes_a549(self):
        completed = self.run_script('--config-dir', 'cross_cell_line', '--list-targets')

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('Multi_5_to_A549', completed.stdout)

    def test_device_override_is_forwarded_to_training_command(self):
        completed = self.run_script('--dry-run', '--device', '0')

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('--device 0', completed.stdout)

    def test_cell_line_specific_config_dir_is_resolved(self):
        completed = self.run_script('--dry-run', '--config-dir', 'cell_line_specific/random', '--target', 'A549')

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('configs/cell_line_specific/random/protocol.yaml', completed.stdout)
        self.assertIn('configs/cell_line_specific/random/A549.yaml', completed.stdout)

    def test_legacy_specific_alias_still_resolves(self):
        completed = self.run_script('--dry-run', '--config-dir', 'specific/random', '--target', 'A549')

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('configs/cell_line_specific/random/protocol.yaml', completed.stdout)
        self.assertIn('configs/cell_line_specific/random/A549.yaml', completed.stdout)

    def test_ambiguous_target_returns_error(self):
        completed = self.run_script('--config-dir', 'recom_sl_partner', '--target', 'A549', '--dry-run')

        self.assertEqual(completed.returncode, 1)
        self.assertIn('ambiguous', completed.stderr.lower())
        self.assertIn('A549_KRAS.yaml', completed.stderr)
        self.assertIn('A549_TP53.yaml', completed.stderr)

    def test_missing_option_value_returns_readable_error(self):
        completed = self.run_script('--target')

        self.assertEqual(completed.returncode, 1)
        self.assertIn('--target requires a value', completed.stderr)
        self.assertIn('Usage: bash scripts/run_mit4sl.sh [options]', completed.stderr)


if __name__ == '__main__':
    unittest.main()
