# 📚 Bookshelf Catalog

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge\&logo=python)
![Flask](https://img.shields.io/badge/Flask-Framework-black?style=for-the-badge\&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge\&logo=opencv)
![Roboflow](https://img.shields.io/badge/Roboflow-Inference%20API-orange?style=for-the-badge)

### Event-Driven Computer Vision Platform for Bookshelf Digitization & Spatial Re-Localization

A full-stack computer vision system engineered to digitize, segment, track, and re-localize book spines across temporal image states using deep learning inference and classical feature-matching pipelines.

</div>

---

# Overview

Bookshelf Catalog is a hybrid deep-learning + classical computer vision platform designed to detect structural changes inside bookshelf configurations.

The system performs:

* 📖 Book spine detection & segmentation
* 🧠 Spatial coordinate tracking
* 🔄 Real-time object re-localization
* ⚡ Multi-stage feature matching
* 🧵 Concurrent asynchronous processing
* 💾 Deterministic cache acceleration

The architecture combines:

| Layer              | Technology             |
| ------------------ | ---------------------- |
| Backend API        | Flask                  |
| Computer Vision    | OpenCV                 |
| Object Detection   | Roboflow Inference API |
| Concurrency        | ThreadPoolExecutor     |
| State Persistence  | SHA-256 + Pickle Cache |
| Feature Matching   | ORB + SIFT             |
| Histogram Analysis | Bhattacharyya Distance |

---

# 🧠 Core Architecture

```text
[Web UI Multipart Stream]
               │
               ▼
[L-Channel Equalization Pipeline]
               │
               ▼
[Dynamic Aspect Ratio Resizer]
               │
               ▼
[SHA-256 Image Fingerprinting]
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
   [Cache Hit]    [Cache Miss]
        │             │
        ▼             ▼
[Coordinate Load]  [Roboflow Inference]
        │             │
        └──────┬──────┘
               ▼
     [Inverse Scaling Transform]
               │
               ▼
[Interactive Boundary Rendering UI]
               │
               ▼
[User Target Selection]
               │
               ▼
[ROI Cross-Matching Engine]
               │
        ┌──────┴──────┐
        ▼             ▼
   [ORB Match]    [SIFT Fallback]
        │             │
        └──────┬──────┘
               ▼
      [Final Object Re-Localization]
```

---

# 🔬 Algorithmic Deep Dive

# 1️⃣ Dynamic Luminance Normalization

To stabilize inference performance under inconsistent lighting conditions, images are normalized using the perceptually uniform **CIE L*a*b*** color space.

## Pipeline

### • Color Space Transformation

The input image is converted from **BGR → LAB** to isolate luminance:

* **L*** → Lightness
* **a*** → Green ↔ Red spectrum
* **b*** → Blue ↔ Yellow spectrum

This separation allows brightness correction without altering chromatic structure.

---

### • Adaptive Brightness Correction

The global luminance mean is calculated:

μ = mean(L*)

If:

μ < 90   → underexposed image
μ > 150  → overexposed image

Then a correction delta is then applied:

ΔL = 120 - μ

Final normalized luminance:

*adjusted = min(max(L* + ΔL, 0), 255)

---

### • Reconstruction

The corrected luminance channel is merged back with chromatic channels and transformed back to BGR space.

✅ Preserves color integrity
✅ Stabilizes inference confidence
✅ Reduces illumination variance

---

# 2️⃣ Cascading ROI Cross-Matching Engine

The re-localization engine uses a cascading multi-stage validation pipeline to minimize computational overhead while maximizing matching accuracy.

---

## 🔹 Tier 1 — Bhattacharyya Histogram Filter

Fast probabilistic filtering based on normalized color distribution similarity.

### Properties

* 3D color histogram comparison
* O(N) computational complexity
* Early rejection optimization

### Validation Condition

```text
Bhattacharyya Distance <= 0.6
```

If failed:

```text
→ Immediate Fast-Reject
```

---

## 🔹 Tier 2 — ORB Feature Alignment

Primary feature-matching layer using:

* FAST keypoint detection
* Rotated BRIEF descriptors
* Hamming-distance brute-force matching

### Complexity

```text
O(K log K)
```

### Validation Rule

```text
Good Matches >= 10
```

If successful:

```text
→ Target Match Registered
```

Otherwise:

```text
→ Escalate to SIFT Fallback
```

---

## 🔹 Tier 3 — SIFT Fallback Engine

High-precision fallback matcher using:

* Difference of Gaussian (DoG)
* Scale-space extrema detection
* 128D orientation descriptors
* FLANN KD-Tree nearest-neighbor search

### Lowe's Ratio Test

To eliminate ambiguous matches:

d1 / d2 < 0.7

Where:

d1 → nearest neighbor distance
d2 → second nearest neighbor distance

✅ Reduces false positives
✅ Improves structural consistency
✅ Preserves spatial reliability

---

# ⚡ Concurrency & Memory Optimization

## 🧵 Asynchronous Parallelism

The system distributes computationally expensive matching operations using:

```python
ThreadPoolExecutor
```

Benefits:

* Parallel ROI processing
* Reduced blocking latency
* Improved throughput under dense feature maps

---

## 💾 Deterministic Cache Layer

Inference responses are cached using:

```text
SHA-256(image_binary_payload)
```

Stored data includes:

* Bounding boxes
* Detection metadata
* Spatial coordinates

Serialized via:

```python
pickle
```

---

## 🔒 Thread-Safe Serialization

Concurrent disk access is protected with:

```python
threading.Lock()
```

Ensuring:

* Atomic write operations
* Consistent cache states
* Race-condition prevention

---

# 🛠 Tech Stack

| Category          | Technology         |
| ----------------- | ------------------ |
| Backend           | Flask              |
| Vision Processing | OpenCV             |
| Detection API     | Roboflow           |
| Concurrency       | concurrent.futures |
| Hashing           | hashlib            |
| Serialization     | pickle             |
| Synchronization   | threading.Lock     |
| Feature Detection | ORB / SIFT         |

---

# 📈 Performance Strategy

The platform prioritizes computational efficiency through:

* Cascading validation filters
* Early candidate rejection
* Multi-threaded feature analysis
* Persistent inference caching
* Resolution-aware scaling
* Adaptive luminance normalization

---

# 🎯 Key Capabilities

✅ Real-time bookshelf tracking
✅ Deep-learning assisted localization
✅ Robust low-light compensation
✅ High-dimensional feature matching
✅ Concurrent asynchronous execution
✅ Deterministic state caching
✅ Structural object re-identification

---
