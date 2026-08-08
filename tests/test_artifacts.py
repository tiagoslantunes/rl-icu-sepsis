"""Dependency-free checks over the committed result artifacts.

These run without gymnasium or stable-baselines3 so CI can verify that the
reported numbers stay internally consistent without retraining any agent.
"""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


class ConfigAResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = load("configA_results.json")

    def test_expected_policies_are_present(self):
        for policy in ("random", "policy_iteration", "q_learning"):
            self.assertIn(policy, self.results)

    def test_metrics_are_in_valid_ranges(self):
        for name, metrics in self.results.items():
            with self.subTest(policy=name):
                self.assertGreaterEqual(metrics["survival_rate"], 0.0)
                self.assertLessEqual(metrics["survival_rate"], 1.0)
                self.assertGreaterEqual(metrics["mean_intensity"], 0.0)
                self.assertGreater(metrics["mean_ep_length"], 0.0)

    def test_policy_iteration_is_the_tabular_optimum(self):
        optimum = self.results["policy_iteration"]["mean_return"]
        for name, metrics in self.results.items():
            if name == "policy_iteration":
                continue
            with self.subTest(policy=name):
                self.assertLessEqual(metrics["mean_return"], optimum)


class ConfigBResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = load("configB_1m_results.json")

    def test_metadata_documents_the_training_budget(self):
        metadata = self.results["metadata"]
        self.assertEqual(metadata["timesteps"], 1_000_000)
        self.assertIn("shaping", metadata)

    def test_reference_policies_are_present(self):
        for policy in ("random", "expert"):
            self.assertIn(policy, self.results)

    def test_survival_rates_are_probabilities(self):
        for name, buckets in self.results.items():
            if name == "metadata" or not isinstance(buckets, dict):
                continue
            for bucket, metrics in buckets.items():
                if not isinstance(metrics, dict) or "survival" not in metrics:
                    continue
                with self.subTest(policy=name, bucket=bucket):
                    self.assertGreaterEqual(metrics["survival"], 0.0)
                    self.assertLessEqual(metrics["survival"], 1.0)


class OverestimationDiagnosticTests(unittest.TestCase):
    def test_bias_matches_its_components(self):
        diagnostic = load("configB_overestimation.json")
        expected = diagnostic["mean_predicted_q"] - diagnostic["mean_realised_return"]
        self.assertAlmostEqual(diagnostic["bias"], expected, places=9)


class ComparisonTableTests(unittest.TestCase):
    def test_header_and_rows_are_well_formed(self):
        lines = (ROOT / "configB_1m_compare_configs.csv").read_text(
            encoding="utf-8"
        ).strip().splitlines()
        self.assertEqual(lines[0], "Config,Agent,Return,Survival,Intensity")
        self.assertGreater(len(lines), 1)
        for line in lines[1:]:
            with self.subTest(row=line):
                self.assertEqual(len(line.split(",")), 5)


if __name__ == "__main__":
    unittest.main()
