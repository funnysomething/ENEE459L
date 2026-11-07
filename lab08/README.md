# Lab 08 — Face detection

This laboratory exercise focuses on building an end-to-end edge AI vision pipeline on resource-constrained embedded hardware, specifically the NVIDIA Jetson Nano. Rather than relying on high-level, black-box vision libraries, this lab emphasizes understanding and implementing both low-level algorithmic image processing (such as custom Canny edge detection from scratch) and deep-learning-based computer vision (using lightweight ONNX neural network inference for real-time face detection).

## What is this lab is built around

This lab is built around two primary technical foundations:

- Hardware-Constrained Edge Computing: Operating within the computational and memory limits of the NVIDIA Jetson Nano (2GB/4GB shared RAM, Maxwell GPU architecture) without causing out-of-memory (OOM) crashes or excessive pipeline latency.

- Dual-Paradigm Computer Vision:
    1. Classical Pixel-Level Image Processing: Implementing the foundational mathematics of gradient computation, spatial filtering, non-maximum suppression, and hysteresis thresholding using pure NumPy/SciPy rather than built-in OpenCV functions.

    2. Modern Embedded Neural Network Inference: Running optimized lightweight deep neural networks (ONNX Runtime, RFB-320) paired with custom post-processing algorithms (vectorized Non-Maximum Suppression and IoU calculations) to locate faces in arbitrary scenes.


## What You Will Build

Throughout this lab, you will build:

- `A Custom Non-Maximum Suppression (NMS) Engine`: A pure NumPy post-processing module that calculates Intersection over Union (IoU) across hundreds of candidate bounding boxes to discard redundant detections and retain only the highest-scoring bounding boxes.

- `An ONNX Deep Learning Face Detection Pipeline`: A complete detection routine using Pillow and ONNX Runtime that ingests raw images, handles spatial resizing, tensor normalization ($[-1.0, 1.0]$), dimension permutation, neural network forward passes, coordinate de-normalization, and visual annotation.

- `A Pure Mathematical Canny Edge Detector`: A multi-stage edge detection pipeline built without OpenCV, comprising:
    1. Gaussian smoothing and dual-axis Sobel kernel convolutions for spatial gradients.
    2. Directional Non-Maximum Suppression to thin raw, blurred gradient boundaries down to crisp 1-pixel-wide lines.
    3. Dual-threshold hysteresis edge tracking to suppress ambient noise while preserving true structural contours.


## How to Write code

The instructions for writing the code are provided in slides in `Module 1` on ELMS. The slide numbers for each of the function are mentioned below -

1. **hard_nms**: pages 1-2
2. **detect_faces**: page 2
3. **non_maximum_suppression**: page 3
4. **hysteresis_thresholding**: pages 3-4.
5. **canny_scipy**: page 4-5

## How to clone lab07 code

```
git remote add upstream https://github.com/YOUR_USERNAME/YOUR_REPO.git
git pull upstream main
git push origin main
```

## How to Run

```bash
cd lab08/

# Install dependencies in your Python environment.
python -m pip install -r requirements.txt

# Generate face annotations and the edge map.
python detect_face.py
```

Running `python lab08/detect_face.py` from the repository root also works.
The bundled model and input image are resolved relative to the script. The
entry point writes `lab08/output_detected.png` and `lab08/edges_scipy.png`.
The functions also accept other input and output paths and return their
detections or edge arrays for further analysis.

The face pipeline resizes RGB input to 320×240 with bicubic interpolation,
normalizes with `(pixel - 127) / 128`, and supplies a contiguous FP32 NCHW
tensor to RFB-320. It filters face scores above 0.7, applies hard NMS at IoU
0.3 to the 200 highest-scoring candidates, and maps boxes back to the original
image. CUDA is selected when the installed ONNX Runtime provides it; otherwise
the pipeline uses the CPU. A Jetson CUDA-enabled runtime must be installed
separately to enable GPU inference.

The edge pipeline converts to grayscale, smooths with a Gaussian at sigma 1.4,
computes Sobel gradients, and scales their magnitude to [0, 255]. Custom
four-direction suppression uses gradient angles in radians. Hysteresis keeps
responses at least 60 and recursively tracks all 8-connected responses at
least 20. Suppressed zero-valued pixels remain background. Constant images
produce an empty edge map without division by zero. No OpenCV functions are
used.

### Verification and supplied samples

From the repository root, run:

```bash
python -m unittest discover -s lab08 -v
```

The nine tests cover overlapping and touching boxes, NMS limits and empty
inputs, zero-area boxes, all four gradient directions, connected weak chains
in both scan directions, threshold boundaries, constant images, real ONNX
inference on the supplied photo, and a no-detection result. Source images are
left unchanged.

The supplied photo produces one face with confidence approximately 0.99986
and integer bounds `(828, 225, 1140, 753)`, matching the sample's green box.
Default font rendering can vary between Pillow versions.

The generated edge map contains **24,010 edge pixels**, including all **16,508**
pixels in `sample_edges_scipy.png`. The reference can be reproduced exactly
by a single top-to-bottom, left-to-right hysteresis scan that discards weak
pixels before a later strong neighbor is visited. This implementation uses
full edge tracking instead, retaining an additional **7,502** connected weak
pixels regardless of scan direction. This difference is intentional and is
covered by a test with a strong seed at the end of a weak diagonal chain.
Validation was performed with CPU inference; Jetson GPU timing was not measured.

## How to Debug your code

There is only one way to debug code - 
1. One way is to use breakpoints. For that, we use `pdb` the package and `pdb.set_trace()` to add a breakpoint at any line of code. 

## How to save your work

Save this work on the existing `solution` branch:
```bash
git switch solution
```

After this, push your changes with following set of commands - 
```bash
git add lab08/
git commit -m "Complete lab08 face detection and Canny edge pipeline"
git push origin solution
```

## Before you hand in
Run the code and tests, and compare the generated images with the supplied
samples, accounting for the hysteresis difference described above. Push the
completed work to `solution`. Before submitting the branch link on Canvas,
verify that it contains the code and generated outputs.
