"""Run from the repository root: python -m unittest discover -s lab08 -v."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import detect_face as detector


class BoxSuppressionTests(unittest.TestCase):
    def setUp(self):
        self.boxes = np.array([
            [0, 0, 10, 10, 0.9],
            [1, 1, 11, 11, 0.8],  # IoU 81/119: suppress this duplicate.
            [20, 20, 30, 30, 0.7],
            [10, 0, 20, 10, 0.6],  # Touching is not overlapping.
        ], dtype=np.float32)

    def test_overlap_score_order_and_input_preservation(self):
        original = self.boxes.copy()
        result = detector.hard_nms(self.boxes, 0.3)
        np.testing.assert_array_equal(result, self.boxes[[0, 2, 3]])
        np.testing.assert_array_equal(self.boxes, original)

    def test_limits_and_empty_input(self):
        np.testing.assert_array_equal(
            detector.hard_nms(self.boxes, 0.3, top_k=1), self.boxes[[0]])
        np.testing.assert_array_equal(
            detector.hard_nms(self.boxes, 0.3, candidate_size=2), self.boxes[[0]])
        self.assertEqual(detector.hard_nms(np.empty((0, 5)), 0.3).shape, (0, 5))

    def test_degenerate_boxes_do_not_divide_by_zero(self):
        boxes = np.array([[0, 0, 0, 0, 0.9], [0, 0, 0, 0, 0.8]])
        with np.errstate(all="raise"):
            self.assertEqual(len(detector.hard_nms(boxes, 0.3)), 2)


class EdgeStageTests(unittest.TestCase):
    def test_all_four_gradient_directions(self):
        # One stronger neighbor along the gradient must suppress the center.
        cases = [(0, (2, 3)), (45, (1, 3)),
                 (90, (1, 2)), (135, (1, 1)), (-45, (1, 1))]
        for degrees, neighbor in cases:
            with self.subTest(degrees=degrees):
                magnitude = np.zeros((5, 5), dtype=np.float32)
                magnitude[2, 2] = 5
                magnitude[neighbor] = 6
                angles = np.full((5, 5), np.deg2rad(degrees))
                result = detector.non_maximum_suppression(magnitude, angles)
                self.assertEqual(result[2, 2], 0)
                magnitude[neighbor] = 4
                result = detector.non_maximum_suppression(magnitude, angles)
                self.assertEqual(result[2, 2], 5)
                self.assertFalse(np.any(result[[0, -1]]))
                self.assertFalse(np.any(result[:, [0, -1]]))

    def test_hysteresis_tracks_entire_chain_in_both_directions(self):
        # A strong seed at the end must recover earlier weak pixels, too.
        image = np.zeros((8, 8), dtype=np.float32)
        np.fill_diagonal(image, 30)
        image[-1, -1] = 80
        image[0, -1] = 30  # Disconnected weak response is discarded.
        expected = np.eye(8, dtype=np.uint8) * 255
        np.testing.assert_array_equal(
            detector.hysteresis_thresholding(image, 20, 60), expected)
        np.testing.assert_array_equal(
            detector.hysteresis_thresholding(image[::-1, ::-1], 20, 60), expected)

    def test_threshold_boundaries_and_no_strong_seeds(self):
        image = np.array([[60, 20, 19, 0]], dtype=float)
        np.testing.assert_array_equal(
            detector.hysteresis_thresholding(image, 20, 60), [[255, 255, 0, 0]])
        self.assertFalse(np.any(detector.hysteresis_thresholding(image, 20, 61)))
        self.assertFalse(np.any(detector.hysteresis_thresholding(
            np.zeros((3, 3)), 0, 0)))

    def test_invalid_inputs(self):
        for low, high in [(-1, 60), (60, 20), (20, np.inf), (np.nan, 60)]:
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                detector.hysteresis_thresholding(np.zeros((3, 3)), low, high)
        with self.assertRaises(ValueError):
            detector.non_maximum_suppression(np.zeros((3, 3)), np.zeros((2, 2)))


class PipelineTests(unittest.TestCase):
    def test_constant_images_have_no_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "flat.png"
            output = Path(directory) / "edges.png"
            for value in (0, 127, 255):
                with self.subTest(value=value):
                    Image.new("L", (32, 32), value).save(source)
                    edges = detector.canny_scipy(source, output)
                    self.assertEqual(edges.dtype, np.uint8)
                    self.assertFalse(np.any(edges))
                    with Image.open(output) as saved:
                        self.assertEqual(saved.mode, "L")
                        self.assertEqual(saved.size, (32, 32))

    def test_supplied_image_and_reference_annotations(self):
        source = detector.LAB_DIR / "face.jpg"
        original = source.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "faces.png"
            detections = detector.detect_faces(source, output)
            self.assertEqual(detections.shape, (1, 5))
            self.assertGreater(detections[0, 4], 0.99)
            np.testing.assert_allclose(detections[0, :4], [828, 225, 1140, 753], atol=1)
            with Image.open(output) as saved:
                self.assertEqual(saved.size, (1920, 1440))
                # Same three-pixel green bounding box as the supplied reference.
                self.assertEqual(saved.getpixel((828, 225)), (0, 255, 0))
            edges = detector.canny_scipy(source, Path(directory) / "edges.png")
            with Image.open(detector.LAB_DIR / "sample_edges_scipy.png") as sample:
                reference = np.asarray(sample)
            # Full hysteresis retains every edge in the sample's forward scan.
            self.assertTrue(np.all(edges[reference > 0] == 255))
            self.assertLess(np.count_nonzero(edges), 0.02 * edges.size)
            self.assertEqual(set(np.unique(edges)), {0, 255})
            no_faces = detector.detect_faces(source, output, conf_threshold=1)
            self.assertEqual(no_faces.shape, (0, 5))
            with Image.open(output) as saved, Image.open(source) as supplied:
                np.testing.assert_array_equal(np.asarray(saved), np.asarray(supplied))
        self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
