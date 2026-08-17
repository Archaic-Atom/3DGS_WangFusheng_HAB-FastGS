import unittest

import torch

from utils.hab_priority_utils import select_guarded_prune_indices


class GuardedPriorityTests(unittest.TestCase):
    def setUp(self):
        self.eligible = torch.ones(8, dtype=torch.bool)
        self.opacity = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.50, 0.60, 0.70, 0.80])

    def test_multiview_only_reorders_low_opacity_band(self):
        score = torch.tensor([0.1, 0.2, 0.9, 0.8, 1.0, 1.0, 1.0, 1.0])
        selected, diagnostics = select_guarded_prune_indices(
            2, self.eligible, self.opacity, score,
            mode="opacity_mv_band", candidate_multiplier=2.0)
        self.assertEqual(set(selected.tolist()), {2, 3})
        self.assertEqual(diagnostics["hab_candidate_band_count"], 4)
        self.assertTrue(all(index < 4 for index in selected.tolist()))

    def test_fisher_guard_protects_sensitive_primitive(self):
        fisher = torch.tensor([100.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        selected, diagnostics = select_guarded_prune_indices(
            2, self.eligible, self.opacity, fisher_proxy=fisher,
            mode="opacity_fisher_guard", fisher_protect_quantile=0.90)
        self.assertEqual(set(selected.tolist()), {1, 2})
        self.assertNotIn(0, selected.tolist())
        self.assertEqual(diagnostics["hab_fisher_protected"], 1)

    def test_guard_relaxes_only_when_exact_count_requires_it(self):
        fisher = torch.arange(1, 9, dtype=torch.float32)
        selected, diagnostics = select_guarded_prune_indices(
            7, self.eligible, self.opacity, fisher_proxy=fisher,
            mode="opacity_fisher_guard", fisher_protect_quantile=0.0)
        self.assertEqual(len(selected), 7)
        self.assertEqual(diagnostics["hab_fisher_guard_relaxed"], 1)
        self.assertNotIn(7, selected.tolist())

    def test_flat_or_missing_multiview_score_falls_back_to_opacity(self):
        flat = torch.ones(8)
        expected = [0, 1]
        for score in (None, flat):
            selected, _ = select_guarded_prune_indices(
                2, self.eligible, self.opacity, score,
                mode="opacity_mv_band", candidate_multiplier=2.0)
            self.assertEqual(selected.tolist(), expected)

    def test_ineligible_primitives_are_never_selected(self):
        eligible = self.eligible.clone()
        eligible[:3] = False
        selected, _ = select_guarded_prune_indices(
            2, eligible, self.opacity, torch.arange(8, dtype=torch.float32),
            mode="opacity_mv_fisher_guard", candidate_multiplier=2.0)
        self.assertTrue(all(index >= 3 for index in selected.tolist()))


if __name__ == "__main__":
    unittest.main()
