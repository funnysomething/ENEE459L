"""Run with: python -m unittest discover -s lab07 -v."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import frames
from schema import BLACK_MAX, FLAT_SPREAD


class PricingTests(unittest.TestCase):
    def test_1080p_reference_costs(self):
        # Independent values supplied by the starter's sample report.
        sizes = [2592000, 2073600, 3110400, 3110400, 6220800, 6220800, 8294400]
        rows = frames.summarise_formats(1920, 1080)
        self.assertEqual([row["format"] for row in rows], list(frames.FORMATS))
        self.assertEqual([row["bytes"] for row in rows], sizes)
        self.assertEqual(rows[2]["relative_to_rgb"], 0.5)
        self.assertEqual(frames.frame_bytes(1920, 1080, "RGB"), 6220800)
        self.assertEqual(frames.bytes_per_second(1920, 1080, "bgr", 30), 186624000)
        self.assertEqual(frames.bytes_per_second(1920, 1080, "nv12", 30), 93312000)
        self.assertEqual(frames.conversion_bytes(1920, 1080, "nv12", "bgr"), 9331200)
        self.assertEqual(frames.resize_bytes(1920, 1080, 224, 224, "rgb"), 6371328)
        self.assertEqual(frames.normalise_bytes(1920, 1080), 31104000)

    def test_small_frame_costs(self):
        self.assertEqual(frames.frame_bytes(4, 2, "raw10"), 10)
        self.assertEqual(frames.frame_bytes(4, 2, "i420"), 12)
        self.assertEqual(frames.bytes_per_second(4, 2, "bgr", 0), 0)
        self.assertEqual(frames.bytes_per_second(4, 2, "bgr", 29.97), 719.28)
        self.assertEqual(frames.normalise_bytes(4, 2, channels=1, out_bytes=2), 24)

    def test_invalid_pricing_parameters(self):
        calls = [
            lambda: frames.frame_bytes(0, 1080, "rgb"),
            lambda: frames.frame_bytes(1920, -1, "rgb"),
            lambda: frames.frame_bytes(1.5, 2, "rgb"),
            lambda: frames.frame_bytes(True, 2, "rgb"),
            lambda: frames.frame_bytes(4, 2, "jpeg"),
            lambda: frames.bytes_per_second(4, 2, "rgb", -1),
            lambda: frames.bytes_per_second(4, 2, "rgb", float("nan")),
            lambda: frames.bytes_per_second(4, 2, "rgb", float("inf")),
            lambda: frames.resize_bytes(4, 2, 0, 224, "rgb"),
            lambda: frames.normalise_bytes(4, 2, channels=0),
            lambda: frames.normalise_bytes(4, 2, out_bytes=-1),
        ]
        for call in calls:
            with self.subTest(call=call), self.assertRaises(frames.FrameError):
                call()


class DiagnosticTests(unittest.TestCase):
    def test_population_statistics_and_generator(self):
        stats = frames.frame_stats(value for value in [0, 0.5, 1])
        self.assertEqual(stats["min"], 0)
        self.assertEqual(stats["max"], 1)
        self.assertEqual(stats["mean"], 0.5)
        self.assertEqual(stats["spread"], 1)
        self.assertAlmostEqual(stats["sd"], (1 / 6) ** 0.5)
        self.assertEqual(stats["samples"], 3)
        self.assertEqual(frames.frame_stats([0.5])["sd"], 0)

    def test_scene_categories_and_threshold_order(self):
        cases = [
            ([0, 0], "black"),
            ([BLACK_MAX, BLACK_MAX], "black"),
            ([0.5, 0.5], "suspicious"),
            ([0.01, 0.01], "suspicious"),
            ([0, FLAT_SPREAD], "dark"),
            ([0, 0.1], "dark"),
            ([0, 0.2], "scene"),
            ([0.2, 0.8], "scene"),
        ]
        for pixels, expected in cases:
            with self.subTest(pixels=pixels):
                label, reason = frames.classify(frames.frame_stats(pixels))
                self.assertEqual(label, expected)
                self.assertTrue(reason)

    def test_invalid_samples(self):
        for pixels in ([], [-0.1], [1.1], [float("nan")], [float("inf")]):
            with self.subTest(pixels=pixels), self.assertRaises(frames.FrameError):
                frames.frame_stats(pixels)


class ImageReportTests(unittest.TestCase):
    def test_supplied_image_without_camera(self):
        source = Path(__file__).with_name("all_focus_s1_1.0_s2_3.0.png")
        original = source.read_bytes()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            frames.cv2, "VideoCapture", side_effect=AssertionError("camera must not open")
        ):
            frames.main(output_dir=directory, input_image=source)
            report = json.loads((Path(directory) / "frame_report.json").read_text())
            self.assertEqual(report["capture_metadata"]["source"], "existing_image")
            self.assertEqual(report["capture_metadata"]["width"], 700)
            self.assertEqual(report["capture_metadata"]["height"], 400)
            self.assertEqual(report["frame_analysis"]["classification"], "scene")
            self.assertEqual(report["frame_analysis"]["stats"]["samples"], 280000)
            pricing = report["memory_pricing"]
            self.assertEqual(pricing["bandwidth_bytes_per_sec"]["bgr_live_stream"], 25200000)
            self.assertEqual(pricing["pipeline_stage_costs"]["nv12_to_bgr_conversion_bytes"], 1260000)
            preview = cv2.imread(str(Path(directory) / "captured_frame.jpg"))
            self.assertEqual(preview.shape, (400, 700, 3))
        self.assertEqual(source.read_bytes(), original)

    def test_one_code_value_image_is_black(self):
        # Float32 normalization can round 1/255 above BLACK_MAX.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "one.png"
            self.assertTrue(cv2.imwrite(str(source), np.ones((2, 4, 3), dtype=np.uint8)))
            frames.main(input_image=source, output_dir=str(Path(directory) / "report"))
            report = json.loads((Path(directory) / "report/frame_report.json").read_text())
            self.assertEqual(report["frame_analysis"]["classification"], "black")

    def test_unreadable_image_does_not_capture(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            frames.cv2, "VideoCapture", side_effect=AssertionError("camera must not open")
        ):
            source = Path(directory) / "invalid.png"
            source.write_text("not an image")
            with self.assertRaises(frames.FrameError):
                frames.main(input_image=source, output_dir=str(Path(directory) / "output"))
            self.assertFalse((Path(directory) / "output").exists())


if __name__ == "__main__":
    unittest.main()
