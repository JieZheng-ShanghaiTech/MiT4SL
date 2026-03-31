import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from util import compute_recommendation_metrics, compute_split_metrics


class RecommendationMetricsTest(unittest.TestCase):
    def test_continuous_recommendation_targets_use_negative_effect_as_relevance(self):
        target = np.array([-2.0, -1.0, 0.5, 0.2])
        scores = np.array([0.9, 0.8, 0.4, 0.1])

        metrics = compute_recommendation_metrics(target, scores, topk_values=(1, 2, 4))

        self.assertAlmostEqual(metrics['ndcg_1'], 1.0)
        self.assertAlmostEqual(metrics['precision_1'], 1.0)
        self.assertAlmostEqual(metrics['ndcg_2'], 1.0)
        self.assertAlmostEqual(metrics['precision_2'], 1.0)
        self.assertAlmostEqual(metrics['precision_4'], 0.5)

    def test_binary_recommendation_targets_are_supported(self):
        target = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.2, 0.8, 0.1])

        metrics = compute_recommendation_metrics(target, scores, topk_values=(1, 2, 3))

        self.assertAlmostEqual(metrics['ndcg_1'], 1.0)
        self.assertAlmostEqual(metrics['precision_1'], 1.0)
        self.assertAlmostEqual(metrics['ndcg_2'], 1.0)
        self.assertAlmostEqual(metrics['precision_2'], 1.0)
        self.assertAlmostEqual(metrics['precision_3'], 2 / 3)

    def test_split_metrics_skip_classification_metrics_for_recommendation_testing(self):
        target = np.array([1, 0, 1, 0])
        logits = torch.tensor(
            [
                [0.1, 2.0],
                [2.0, 0.1],
                [0.2, 1.5],
                [1.2, 0.2],
            ],
            dtype=torch.float32,
        )

        metrics = compute_split_metrics(
            target,
            logits,
            include_classification_metrics=False,
            include_recommendation_metrics=True,
        )

        self.assertNotIn('auc', metrics)
        self.assertNotIn('aupr', metrics)
        self.assertNotIn('bacc', metrics)
        self.assertEqual(
            set(metrics),
            {'ndcg_5', 'precision_5', 'ndcg_10', 'precision_10', 'ndcg_20', 'precision_20'},
        )

    def test_split_metrics_keep_classification_metrics_for_standard_tasks(self):
        target = np.array([1, 0, 1, 0])
        logits = torch.tensor(
            [
                [0.1, 2.0],
                [2.0, 0.1],
                [0.2, 1.5],
                [1.2, 0.2],
            ],
            dtype=torch.float32,
        )

        metrics = compute_split_metrics(
            target,
            logits,
            include_classification_metrics=True,
            include_recommendation_metrics=False,
        )

        self.assertIn('auc', metrics)
        self.assertIn('aupr', metrics)
        self.assertIn('bacc', metrics)


if __name__ == '__main__':
    unittest.main()
