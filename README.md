# 📚 Bookshelf Catalog

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Framework-lightgrey.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Roboflow](https://img.shields.io/badge/Roboflow-Inference%20API-orange.svg)](https://roboflow.com/)

A full-stack, event-driven Computer Vision platform engineered to digitize, segment, and track non-rigid physical configurations of book spines across discrete time intervals. The core system maps spatial coordinate deviations between modified states, executing real-time object re-localization via a cascading heuristic-and-deterministic pixel matching pipeline.

---

## 🔬 Technical Deep Dive & Algorithmic Mechanics

### 1. Dynamic Luminance Normalization (CIE L\*a\*b\* Space)
To stabilize deep-learning object detection against uneven lighting, shadows, and camera exposure shifts, images are processed through a non-destructive adaptive brightness pipeline prior to inference:
* **Color Space Shift:** The image is translated from non-linear $BGR$ to the perceptually uniform $CIE\ L^*a^*b^*$ color space to isolate structural luminance ($L^*$) from chromaticity ($a^*, b^*$).
* **Mathematical Vector Clipping:** The global mean ($\mu_{L^*}$) of the luminance channel is calculated. If $\mu_{L^*}$ drops below the absolute dark barrier ($90$) or exceeds the bright ceiling ($150$), a linear correction delta ($\Delta L = 120 - \mu_{L^*}$) is applied across the matrix:

$$\mathbf{L}^*_{\text{adjusted}} = \min(\max(\mathbf{L}^* + \Delta L, 0), 255)$$

* **Re-merging:** Channels are reconstructed and mapped back to the $BGR$ domain, preserving underlying chromatic definitions while maximizing contrast uniformity.

### 2. Cascading Region-Of-Interest (ROI) Cross-Matching Engine
When an index is targeted in the source frame, the matcher isolates the target sub-matrix and routes candidate coordinates through a descending complexity matrix filter to preserve compute cycles:

[Target Sub-Matrix Node]
│
▼
┌────────────────────────────────────────────────────────┐
│ Tier 1: Bhattacharyya Color Histogram Space Filter     │
│ - Evaluates normalized 3D color distribution matrices  │
│ - Computational Complexity: O(N)                       │
└──────────────────────────┬─────────────────────────────┘
│
Is Distance Metric <= 0.6?
├───► [NO]  ──► Fast-Reject Node
└───► [YES]
│
▼
┌────────────────────────────────────────────────────────┐
│ Tier 2: Parallelized Binary Feature Alignment (ORB)    │
│ - Computes FAST keypoints & modified BRIEF descriptors │
│ - Intensity-weighted centroid orientation tracking     │
│ - Matching Evaluator: Hamming Distance Matrix via Brute│
│ - Operational Complexity: O(K log K)                   │
└──────────────────────────┬─────────────────────────────┘
│
Are Good Matches Count >= 10?
├───► [YES] ──► Target Match Registered
└───► [NO]
│
▼
┌────────────────────────────────────────────────────────┐
│ Tier 3: Gradient Distribution Fallback (SIFT)          │
│ - Scale-space extrema detection via DoG convolution    │
│ - 128-dimensional orientation vector assignment        │
│ - Feature Matching Evaluator: FLANN KD-Tree Search     │
│ - Mathematical Fallback Validation: Lowe's Ratio Test  │
└────────────────────────────────────────────────────────┘


* **Lowe's Ratio Test Validation:** SIFT keypoint matches are verified by evaluating the closest neighbor distance ratio against the second-closest neighbor ($d_1 / d_2 < 0.7$), discarding ambiguous background structures.

### 3. Asynchronous Task Concurrency & Memory Gateways
* **ThreadPool Worker Pool:** To bypass block-stalls during high-dimensional cross-matching across dense arrays ($M \times N$ keypoint combinations), operations are partitioned dynamically across an asynchronous `ThreadPoolExecutor`.
* **State Verification Caching:** Images are uniquely indexed using deterministic `SHA-256` checksums of their binary payloads. Bounding box coordinates returned from Roboflow are serialized via `pickle` into a disk-backed dictionary map.
* **Thread Mutex Lock:** Access to the serialization workspace is isolated via a mutual exclusion primitive (`threading.Lock()`), ensuring atomic write states during concurrent HTTP worker access cycles.

---

## 🛠 Multi-Tier Structural Pipeline

The system coordinates deep learning abstractions, reactive UI binding, and classic computer vision heuristics through a unified processing flow:

```text
[Web UI Multipart Stream] ──► [L-Channel Equalization Shift] ──► [Dynamic Aspect Ratio Shrinkage]
                                                                             │
                                                                             ▼
[Thread-Safe Local Cache] ◄── [Deterministic SHA256 Hashing] ◄───────────────┤
         │
         ├───► [Cache Hit]  ──► Deserialization of Coordinate List (0ms Overhead)
         │
         └───► [Cache Miss] ──► [Roboflow API Inference Topology] ──► [Inverse Scaling Transform]
                                                                             │
                                                                             ▼
[Interactive Web Canvas Interface] ◄─────────────────────────────────────────┘
(Asynchronous Relative Boundary Plotting Engine)
         │
         ▼ [User Selection Pointer Dispatch]
[3D Bhattacharyya Color Histogram Validation Filter]
         │
         ├───► [Passes] ──► [Parallelized ORB Keypoint Cross-Check] ──► Structural Highlight Alignment
         │
         └───► [Fails]  ──► [High-Dimensional SIFT Fallback Engine]  ──► Lowe's Ratio Score Render