from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

LAB_DIR = Path(__file__).resolve().parent
MODEL_PATH = LAB_DIR / "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """Keep high-scoring, non-overlapping rows of [x1, y1, x2, y2, score]."""
    box_scores = np.asarray(box_scores)
    if box_scores.ndim != 2 or box_scores.shape[1] != 5:
        raise ValueError("box_scores must have shape (N, 5)")
    if not 0 <= iou_threshold <= 1 or candidate_size <= 0:
        raise ValueError("IoU must be in [0, 1] and candidate_size must be positive")
    boxes = box_scores[:, :4]
    areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    indices = np.argsort(box_scores[:, 4])[-candidate_size:]
    picked = []
    while indices.size:
        current = indices[-1]
        picked.append(current)
        if (top_k > 0 and len(picked) >= top_k) or indices.size == 1:
            break
        indices = indices[:-1]
        overlap = np.maximum(
            np.minimum(boxes[indices, 2:], boxes[current, 2:])
            - np.maximum(boxes[indices, :2], boxes[current, :2]),
            0,
        )
        intersection = overlap[:, 0] * overlap[:, 1]
        union = areas[indices] + areas[current] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection,
                        dtype=float), where=union > 0)
        indices = indices[iou <= iou_threshold]
    return box_scores[picked]

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    """Run RFB-320, save face annotations, and return pixel boxes with scores."""
    if not 0 <= conf_threshold <= 1:
        raise ValueError("conf_threshold must be in [0, 1]")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    resampling = getattr(Image, "Resampling", Image)
    resized = image.resize((320, 240), resampling.BICUBIC)
    tensor = (np.asarray(resized, dtype=np.float32) - 127.0) / 128.0
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
    session = ort.InferenceSession(
        str(MODEL_PATH), sess_options=sess_options, providers=available_providers
    )
    scores, boxes = session.run(None, {session.get_inputs()[0].name: tensor})
    face_scores = scores[0, :, 1]
    selected = face_scores > conf_threshold
    candidates = np.column_stack((boxes[0, selected], face_scores[selected]))
    detections = hard_nms(candidates, iou_threshold=0.3)
    detections[:, :4] *= np.array([width, height, width, height])
    detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, width - 1)
    detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, height - 1)
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, score in detections:
        bounds = tuple(int(value) for value in (x1, y1, x2, y2))
        draw.rectangle(bounds, outline="lime", width=3)
        draw.text((bounds[0], max(0, bounds[1] - 15)),
                  f"Face: {score:.2f}", fill="lime")
    image.save(output_path)
    print(f"Detected {len(detections)} face(s); saved {output_path}")
    return detections

# Edge detection
def non_maximum_suppression(magnitude, angle):
    """Thin gradients along four directions; angles are in radians."""
    magnitude = np.asarray(magnitude)
    angle = np.asarray(angle)
    if magnitude.ndim != 2 or angle.shape != magnitude.shape:
        raise ValueError("magnitude and angle must be matching 2-D arrays")
    result = np.zeros_like(magnitude)
    degrees = np.rad2deg(angle[1:-1, 1:-1]) % 180
    center = magnitude[1:-1, 1:-1]
    # The vertical Sobel kernel points upward, so 45 degrees compares NE/SW.
    directions = (
        ((degrees < 22.5) | (degrees >= 157.5),
         magnitude[1:-1, :-2], magnitude[1:-1, 2:]),
        ((degrees >= 22.5) & (degrees < 67.5),
         magnitude[:-2, 2:], magnitude[2:, :-2]),
        ((degrees >= 67.5) & (degrees < 112.5),
         magnitude[:-2, 1:-1], magnitude[2:, 1:-1]),
        ((degrees >= 112.5) & (degrees < 157.5),
         magnitude[:-2, :-2], magnitude[2:, 2:]),
    )
    for direction, first, second in directions:
        keep = direction & (center >= first) & (center >= second)
        result[1:-1, 1:-1][keep] = center[keep]
    return result

def hysteresis_thresholding(img, low_thresh, high_thresh):
    """Keep strong edges and all 8-connected weak edges reachable from them."""
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("img must be a 2-D array")
    if not 0 <= low_thresh <= high_thresh or not np.isfinite(high_thresh):
        raise ValueError("thresholds must be finite with 0 <= low <= high")
    # Zero means a suppressed pixel, including when either threshold is zero.
    eligible = (img >= low_thresh) & (img > 0)
    strong = (img >= high_thresh) & eligible
    result = np.zeros(img.shape, dtype=np.uint8)
    result[strong] = 255
    stack = list(zip(*np.nonzero(strong)))
    height, width = img.shape
    while stack:
        y, x = stack.pop()
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                if eligible[ny, nx] and result[ny, nx] == 0:
                    result[ny, nx] = 255
                    stack.append((ny, nx))
    return result

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    """Save a Canny edge map using Gaussian/Sobel filtering and custom tracking."""
    with Image.open(image_path) as source:
        gray = np.asarray(source.convert("L"), dtype=np.float32)
    smoothed = gaussian_filter(gray, sigma=1.4)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = convolve(smoothed, sobel_x)
    gy = convolve(smoothed, sobel_y)
    magnitude = np.hypot(gx, gy)
    maximum = magnitude.max()
    if maximum > 0:
        magnitude *= 255.0 / maximum
    thinned = non_maximum_suppression(magnitude, np.arctan2(gy, gx))
    edges = hysteresis_thresholding(thinned, low, high)
    Image.fromarray(edges).save(output_path)
    print(f"Saved edges to {output_path}")
    return edges

if __name__ == "__main__":
    detect_faces(LAB_DIR / "face.jpg", LAB_DIR / "output_detected.png")
    canny_scipy(LAB_DIR / "face.jpg", LAB_DIR / "edges_scipy.png")
