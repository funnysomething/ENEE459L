from __future__ import annotations

import argparse
import json
import math
import time
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from schema import BLACK_MAX, FLAT_SPREAD


class FrameError(ValueError):
    """An invalid frame shape, pixel sample, or pricing parameter."""


# The formats this lab prices, in the order Lecture 07 slide 27 lists them.
FORMATS: tuple[str, ...] = ("raw10", "gray8", "nv12", "i420", "rgb", "bgr",
                            "bgrx")

# What each format is, in one clause, for the summary table and the error
# message. Not used by any arithmetic.
FORMAT_NOTES: dict[str, str] = {
    "raw10": "on the CSI wire, four pixels packed into five bytes, no ISP yet",
    "gray8": "luma only, which is a decision about colour and not a format",
    "nv12": "luma full, chroma quartered and interleaved -- the ISP's output",
    "i420": "the same bytes as NV12 with the two chroma planes kept apart",
    "rgb": "three bytes, nothing shared, nothing wasted",
    "bgr": "the same three bytes, ordered the way OpenCV expects them",
    "bgrx": "four bytes, one of which nobody reads, bought for alignment",
}


def _bytes_per_pixel(fmt: str) -> float:
    table = {
        # One byte of luma per pixel, plus a quarter of a byte of Cb and a
        # quarter of a byte of Cr, because the chroma plane is half width and
        # half height. NV12 interleaves the two chroma channels in one plane
        # and I420 keeps them in two, which changes the layout and not the
        # total, which is why both entries are the same number.
        "nv12": 1.5,
        "i420": 1.5,
        # Ten bits per pixel on the wire: four pixels in five bytes. This is
        # the only entry describing something that has not reached memory yet.
        "raw10": 1.25,
        "gray8": 1.0,
        "rgb": 3.0,
        "bgr": 3.0,
        # The fourth byte is padding. It carries no information and it costs
        # exactly as much to move as the three that do.
        "bgrx": 4.0,
    }
    key = str(fmt).lower()
    if key not in table:
        raise FrameError(
            f"format {fmt!r} is not one of {', '.join(FORMATS)}. If you meant "
            "a format this lab does not price, add it here and say in your "
            "paragraph what its bytes per pixel is and how you know."
        )
    return table[key]



# ---------------------------------------------------------------------------
# Hardware Capture & Pipeline Configuration
# ---------------------------------------------------------------------------

def gstreamer_pipeline(
    capture_width: int = 1920,
    capture_height: int = 1080,
    display_width: int = 1920,
    display_height: int = 1080,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=1"
    )


def grab_frame(pipeline: str, warmup_frames: int = 15) -> np.ndarray:
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera via GStreamer pipeline.")

    try:
        # Warmup loop to allow sensor auto-exposure and AWB to converge
        for _ in range(warmup_frames):
            ret, _ = cap.read()
            if not ret:
                time.sleep(0.03)

        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("Camera opened, but failed to capture a valid frame.")
        return frame
    finally:
        cap.release()


############################
## Your work starts here ###
############################

def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise FrameError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def frame_bytes(width: int, height: int, fmt: str) -> int:
    """Tightly packed payload size, excluding row alignment and buffer padding."""
    width = _positive_int(width, "width")
    height = _positive_int(height, "height")
    return math.ceil(width * height * _bytes_per_pixel(fmt))


def bytes_per_second(width: int, height: int, fmt: str, fps: float) -> float:
    """One frame's payload transferred at the specified (assumed) frame rate."""
    if not math.isfinite(fps) or fps < 0:
        raise FrameError("fps must be finite and nonnegative")
    return frame_bytes(width, height, fmt) * float(fps)


def frame_stats(pixels: Sequence[float] | Iterable[float]) -> dict[str, float]:
    """Summarise normalized luma using the population standard deviation."""
    values = np.fromiter(pixels, dtype=np.float64)
    if values.size == 0:
        raise FrameError("cannot analyse an empty frame")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise FrameError("luma samples must be finite and in [0, 1]")
    low = float(values.min())
    high = float(values.max())
    return {
        "min": low,
        "max": high,
        "mean": float(values.mean()),
        "spread": high - low,
        "sd": float(values.std()),
        "samples": float(values.size),
    }


def classify(stats: dict[str, float]) -> tuple[str, str]:
    """Distinguish no signal, flat fields, dark scenes, and exposed scenes."""
    if stats["max"] <= BLACK_MAX:
        return "black", f"maximum {stats['max']:.4f} <= {BLACK_MAX:.4f}. No sensor signal."
    if stats["spread"] < FLAT_SPREAD:
        return "suspicious", f"spread {stats['spread']:.4f} < {FLAT_SPREAD:.4f}. Signal without structure."
    if stats["mean"] < 0.10:
        return "dark", f"mean {stats['mean']:.4f} < 0.10. Structured but underexposed."
    return "scene", f"mean {stats['mean']:.4f}, spread {stats['spread']:.4f}. Exposed and structured."


def conversion_bytes(width: int, height: int, src: str, dst: str) -> int:
    """Read the source once and write the destination once."""
    return frame_bytes(width, height, src) + frame_bytes(width, height, dst)


def resize_bytes(src_w: int, src_h: int, dst_w: int, dst_h: int,
                 fmt: str) -> int:
    """Budget one full source read and one resized output write."""
    return frame_bytes(src_w, src_h, fmt) + frame_bytes(dst_w, dst_h, fmt)


def normalise_bytes(width: int, height: int, channels: int = 3,
                    out_bytes: int = 4) -> int:
    """Read uint8 channels and write the normalized tensor (FP32 by default)."""
    width = _positive_int(width, "width")
    height = _positive_int(height, "height")
    channels = _positive_int(channels, "channels")
    out_bytes = _positive_int(out_bytes, "out_bytes")
    return width * height * channels * (1 + out_bytes)


def summarise_formats(width: int, height: int) -> list[dict[str, Any]]:
    rgb_bytes = frame_bytes(width, height, "rgb")
    return [
        {
            "format": fmt,
            "bytes_per_pixel": _bytes_per_pixel(fmt),
            "bytes": frame_bytes(width, height, fmt),
            "relative_to_rgb": frame_bytes(width, height, fmt) / rgb_bytes,
            "note": FORMAT_NOTES[fmt],
        }
        for fmt in FORMATS
    ]


###############################
##### The main function #######
###############################
def main(
    output_dir: str = "output",
    image_filename: str = "captured_frame.jpg",
    json_filename: str = "frame_report.json",
    target_fps: float = 30.0,
    input_image: str | Path | None = None,
) -> None:
    if not math.isfinite(target_fps) or target_fps < 0:
        raise FrameError("fps must be finite and nonnegative")

    source_path = Path(input_image) if input_image is not None else None
    out_path = Path(output_dir)
    img_path = out_path / image_filename
    json_path = out_path / json_filename
    if source_path is not None and source_path.resolve() in (img_path.resolve(), json_path.resolve()):
        raise FrameError("output paths must not overwrite the input image")

    if source_path is None:
        pipeline = gstreamer_pipeline()
        print("Capturing from CSI camera (1920x1080 at 30 FPS)...", flush=True)
        frame_bgr = grab_frame(pipeline)
    else:
        frame_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FrameError(f"could not read input image: {source_path}")
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save the captured frame or a preview of the supplied image.
    if not cv2.imwrite(str(img_path), frame_bgr):
        raise FrameError(f"could not save image: {img_path}")
    height, width, _ = frame_bgr.shape

    # 2. Extract normalized luma/grayscale values in [0.0, 1.0] for classification
    # classify() tests against BLACK_MAX (1/255 ≈ 0.0039) and mean < 0.10
    gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    normalized_luma = gray_frame.astype(np.float64).ravel() / 255.0

    # 3. Compute statistics and scene category
    stats = frame_stats(normalized_luma)
    classification, reason = classify(stats)

    # 4. Calculate memory bus bandwidth and conversion costs
    formats_table = summarise_formats(width, height)
    bgr_bandwidth_bps = bytes_per_second(width, height, "bgr", target_fps)
    nv12_to_bgr_cost = conversion_bytes(width, height, "nv12", "bgr")
    norm_cost = normalise_bytes(width, height, channels=3, out_bytes=4)

    # Downsampling budget from the image's actual size to a 224x224 tensor.
    resize_cost = resize_bytes(width, height, 224, 224, "rgb")

    # 5. Aggregate payload and write JSON
    payload: dict[str, Any] = {
        "capture_metadata": {
            "source": "csi_camera" if source_path is None else "existing_image",
            **({"pipeline": pipeline, "configured_capture_fps": 30}
               if source_path is None else {"source_image_path": str(source_path.resolve())}),
            "saved_image_path": str(img_path.resolve()),
            "width": width,
            "height": height,
            "channels": 3,
            "assumed_fps": target_fps,
        },
        "frame_analysis": {
            "classification": classification,
            "classification_reason": reason,
            "stats": stats,
        },
        "memory_pricing": {
            "assumptions": {
                "storage": "tightly packed payloads; excludes stride and buffer padding",
                "stage_costs": "one full input read plus one output write per stage",
                "fps": "assumed for budgeting; not measured from a single frame",
                "normalization": "full-resolution uint8 input to FP32 output",
            },
            "bandwidth_bytes_per_sec": {
                "bgr_live_stream": bgr_bandwidth_bps,
                "nv12_live_stream": bytes_per_second(width, height, "nv12", target_fps),
            },
            "format_comparison": formats_table,
            "pipeline_stage_costs": {
                "nv12_to_bgr_conversion_bytes": nv12_to_bgr_cost,
                "resize_to_224x224_rgb_bytes": resize_cost,
                "fp32_normalization_tensor_bytes": norm_cost,
            },
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")

    print(f"Input analysed          : {source_path if source_path is not None else 'CSI camera'}")
    print(f"Frame preview saved to  : {img_path}")
    print(f"Analysis saved to       : {json_path}")
    print(f"Classification result   : {classification.upper()} ({reason})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture a CSI camera frame and price frame memory traffic.")
    parser.add_argument("--input", type=Path,
                        help="analyse an existing image instead of capturing from the camera")
    parser.add_argument("--output-dir", default=str(Path(__file__).with_name("output")))
    parser.add_argument("--fps", type=float, default=30.0, help="assumed rate for bandwidth budgeting")
    args = parser.parse_args()
    try:
        main(input_image=args.input, output_dir=args.output_dir, target_fps=args.fps)
    except (FrameError, OSError, RuntimeError, cv2.error) as exc:
        parser.exit(3, f"error: {exc}\n")
