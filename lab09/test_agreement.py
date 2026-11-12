"""Run from the repository root: python -m unittest discover -s lab09 -v."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import agreement as metrics


class LatencyTests(unittest.TestCase):
    def test_interpolated_percentiles_and_generator(self):
        report = metrics.latency_report(iter([4, 0, 2, 1]))
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["mean"], 1.75)
        self.assertEqual(report["p50"], 1.5)
        self.assertAlmostEqual(report["p95"], 3.7)
        self.assertAlmostEqual(report["p99"], 3.94)
        self.assertEqual(report["worst"], 4)
        self.assertAlmostEqual(report["jitter"], 2.2)

    def test_single_zero_sample(self):
        self.assertEqual(metrics.latency_report([0]), {
            "n": 1, "mean": 0, "worst": 0,
            "p50": 0, "p95": 0, "p99": 0, "jitter": 0})

    def test_speedup_includes_slowdowns_and_equal_durations(self):
        self.assertEqual(metrics.speedup(8, 2),
                         {"factor": 4, "faster": True, "saved_ms": 6})
        self.assertEqual(metrics.speedup(2, 8),
                         {"factor": 0.25, "faster": False, "saved_ms": -6})
        self.assertFalse(metrics.speedup(2, 2)["faster"])
        self.assertEqual(metrics.speedup(0, 2)["factor"], 0)


class DeviationTests(unittest.TestCase):
    def test_absolute_deviation_ties_use_first_index(self):
        self.assertEqual(metrics.max_abs_dev([-2, 4, 1], [0, 2, 1]), {
            "max_abs": 2, "index": 0, "a": -2, "b": 0, "n": 3})
        self.assertEqual(metrics.max_abs_dev([1, 1], [1, 1])["index"], 0)

    def test_relative_reference_floor_and_ties(self):
        result = metrics.max_rel_dev([1, -2, 4, 2], [0, -1, 2, 2], floor=1)
        self.assertEqual(result, {"max_rel": 1, "index": 0,
                                 "floored": 1, "denom_min": 0, "n": 4})
        self.assertEqual(metrics.max_rel_dev([1], [2])["max_rel"], 0.5)

    def test_floor_disclosure(self):
        absolute = metrics.max_abs_dev([0.001], [0])
        relative = metrics.max_rel_dev([0.001], [0])
        self.assertIn("denominator floored on 1 of 1 entries",
                      metrics.deviation_sentence(absolute, relative))


class DecisionTests(unittest.TestCase):
    def test_argmax_ties_and_first_disagreement(self):
        self.assertEqual(metrics.top1_agreement(
            [[2, 2], [0, 1], [2, 1]], [[2, 1], [1, 0], [1, 2]]), {
                "n": 3, "agree": 1, "disagree": 2,
                "rate": 1 / 3, "first_disagreement": 1})
        self.assertIsNone(metrics.top1_agreement([[0]], [[0]])[
            "first_disagreement"])

    def test_margin_statistics_and_first_minimum(self):
        self.assertEqual(metrics.decision_margin(
            [[4, 0, 1], [-2, -1, -4], [0, 1]], tol=0.1), {
                "n": 3, "min": 1, "median": 1, "mean": 5 / 3,
                "argmin": 1, "below_tol": 0, "safe": True})
        self.assertEqual(metrics.decision_margin(
            [[4, 1], [2, 1]], 0)["median"], 2)

    def test_margin_accounts_for_both_scores_moving(self):
        # Both score errors are <= 1, yet a margin > 1 can still flip.
        self.assertFalse(metrics.decision_margin([[1.5, 0]], 1)["safe"])
        self.assertEqual(metrics.top1_agreement(
            [[1.5, 0]], [[0.5, 1]])["disagree"], 1)
        self.assertFalse(metrics.decision_margin([[2, 0]], 1)["safe"])
        self.assertTrue(metrics.decision_margin([[2.01, 0]], 1)["safe"])
        self.assertFalse(metrics.decision_margin([[0, 0]], 0)["safe"])


class ValidationTests(unittest.TestCase):
    def test_rejects_nonfinite_and_nonnumeric_values_everywhere(self):
        for bad in [float("nan"), float("inf"), -float("inf"), True, "1", None]:
            calls = [
                lambda: metrics.latency_report([1, bad]),
                lambda: metrics.speedup(bad, 1),
                lambda: metrics.speedup(1, bad),
                lambda: metrics.max_abs_dev([0, bad], [1, 1]),
                lambda: metrics.max_abs_dev([1, 1], [0, bad]),
                lambda: metrics.max_rel_dev([0, bad], [1, 1]),
                lambda: metrics.max_rel_dev([1, 1], [0, bad]),
                lambda: metrics.max_rel_dev([1], [1], floor=bad),
                lambda: metrics.top1_agreement([[0, bad]], [[1, 0]]),
                lambda: metrics.top1_agreement([[1, 0]], [[0, bad]]),
                lambda: metrics.decision_margin([[0, bad]], 1),
                lambda: metrics.decision_margin([[0, 1]], bad),
            ]
            for i, call in enumerate(calls):
                with self.subTest(value=bad, call=i):
                    with self.assertRaises(metrics.AgreementError):
                        call()

    def test_rejects_empty_mismatched_or_invalid_domains(self):
        calls = [
            lambda: metrics.latency_report([]),
            lambda: metrics.latency_report([-1]),
            lambda: metrics.speedup(-1, 1),
            lambda: metrics.speedup(1, 0),
            lambda: metrics.speedup(1, -1),
            lambda: metrics.max_abs_dev([], []),
            lambda: metrics.max_abs_dev([1], [1, 2]),
            lambda: metrics.max_rel_dev([], []),
            lambda: metrics.max_rel_dev([1], []),
            lambda: metrics.max_rel_dev([1], [1], floor=0),
            lambda: metrics.max_rel_dev([1], [1], floor=-1),
            lambda: metrics.top1_agreement([], []),
            lambda: metrics.top1_agreement([[1]], []),
            lambda: metrics.top1_agreement([[]], [[]]),
            lambda: metrics.top1_agreement([[1, 2]], [[1]]),
            lambda: metrics.decision_margin([], 0),
            lambda: metrics.decision_margin([[]], 0),
            lambda: metrics.decision_margin([[1]], 0),
            lambda: metrics.decision_margin([[1, 0]], -1),
        ]
        for i, call in enumerate(calls):
            with self.subTest(call=i), self.assertRaises(metrics.AgreementError):
                call()


class SampleTests(unittest.TestCase):
    def test_full_report_matches_supplied_sample(self):
        lab = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with redirect_stdout(io.StringIO()):
                metrics.main(lab / "benchmark_input_data.json", output)
            self.assertEqual(json.loads(output.read_text()), json.loads(
                (lab / "sample_benchmark_results.json").read_text()))


if __name__ == "__main__":
    unittest.main()
