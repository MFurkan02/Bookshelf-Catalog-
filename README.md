# 📚 Bookshelf Catalog

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge\&logo=python)
![Flask](https://img.shields.io/badge/Flask-Framework-black?style=for-the-badge\&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge\&logo=opencv)
![Roboflow](https://img.shields.io/badge/Roboflow-Inference%20API-orange?style=for-the-badge)

### Event-Driven Computer Vision Platform for Bookshelf Digitization & Spatial Re-Localization

A hybrid deep-learning and classical computer vision platform engineered to detect, segment, track, and re-localize book spines across dynamic bookshelf states using feature-matching pipelines and AI-powered object detection.

</div>

---

# 📚 Table of Contents

* [Overview](#-overview)
* [Core Features](#-core-features)
* [System Architecture](#-system-architecture)
* [Algorithmic Deep Dive](#-algorithmic-deep-dive)
* [Concurrency & Memory Optimization](#-concurrency--memory-optimization)
* [Technology Stack](#-technology-stack)
* [Performance Strategy](#-performance-strategy)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Running the Application](#-running-the-application)
* [Future Improvements](#-future-improvements)
* [License](#-license)

---

# 📌 Overview

Bookshelf Catalog is a full-stack computer vision system designed to monitor structural changes inside bookshelf environments through AI-assisted object localization and feature-based re-identification.

The platform combines:

* Deep-learning inference pipelines
* Classical feature-matching algorithms
* Concurrent asynchronous processing
* Deterministic caching mechanisms
* Spatial coordinate reconstruction

The system is optimized for:

* Bookshelf digitization
* Object re-localization
* Temporal scene comparison
* Inventory tracking
* Computer vision experimentation

---

# ✨ Core Features

## 📖 Book Spine Detection & Segmentation

The system performs automated detection and extraction of book spine regions using the Roboflow Inference API.

Capabilities include:

* Bounding-box detection
* Spatial coordinate extraction
* Dynamic region segmentation
* Resolution-aware scaling

---

## 🧠 Multi-Stage Re-Localization Engine

Books are re-identified across temporal image states using a cascading validation architecture.

Matching pipeline:

1. Histogram similarity filtering
2. ORB feature alignment
3. SIFT fallback validation

This design minimizes computational overhead while maximizing structural consistency.

---

## ⚡ Concurrent Asynchronous Processing

The application distributes computationally intensive matching operations through parallel worker execution using:

```python id="a8x91d"
ThreadPoolExecutor
```

Benefits:

* Reduced blocking latency
* Parallel ROI analysis
* Higher throughput under dense feature maps

---

## 💾 Deterministic Cache Acceleration

Inference outputs are cached using SHA-256 image fingerprinting.

Stored metadata includes:

* Bounding boxes
* Spatial coordinates
* Detection metadata
* Feature extraction states

This significantly reduces redundant inference requests and accelerates repeated processing.

---

## 🔄 Real-Time Object Re-Localization

The system tracks structural object movement between image states and reconstructs probable object positions using geometric feature analysis.

Supported operations:

* Target selection
* ROI matching
* Feature validation
* Positional reconstruction

---

# 🏗️ System Architecture

```text id="i7u5m2"
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

To stabilize inference performance under inconsistent illumination conditions, the platform normalizes images using the perceptually uniform **CIE L*a*b*** color space.

---

## 🔹 Color Space Transformation

The image is transformed from:

```text id="uv5f5r"
BGR → LAB
```

Channel decomposition:

| Channel | Description            |
| ------- | ---------------------- |
| L*      | Lightness              |
| a*      | Green ↔ Red spectrum   |
| b*      | Blue ↔ Yellow spectrum |

This separation enables luminance correction while preserving chromatic integrity.

---

## 🔹 Adaptive Brightness Correction

The global luminance mean is calculated:

```text id="f9p4xh"
μ = mean(L*)
```

Correction thresholds:

```text id="zj6f2n"
μ < 90   → Underexposed Image
μ > 150  → Overexposed Image
```

Brightness adjustment:

```text id="mgv0qh"
ΔL = 120 - μ
```

Normalized luminance:

```text id="kwm4rk"
adjusted = min(max(L* + ΔL, 0), 255)
```

### Benefits

✅ Preserves color consistency
✅ Stabilizes detection confidence
✅ Reduces illumination variance

---

# 2️⃣ Cascading ROI Cross-Matching Engine

The re-localization pipeline uses a multi-stage validation strategy to reduce false positives while preserving computational efficiency.

---

## 🔹 Tier 1 — Bhattacharyya Histogram Filter

Fast probabilistic filtering based on normalized color distribution similarity.

### Features

* 3D histogram comparison
* Early rejection optimization
* Low computational complexity

### Validation Rule

```text id="n5v1ha"
Bhattacharyya Distance <= 0.6
```

If validation fails:

```text id="l2g0qo"
→ Immediate Fast-Reject
```

---

## 🔹 Tier 2 — ORB Feature Alignment

Primary feature matching layer using:

* FAST keypoint detection
* Rotated BRIEF descriptors
* Hamming-distance matching

### Computational Complexity

```text id="cs5rwd"
O(K log K)
```

### Validation Rule

```text id="o3j0et"
Good Matches >= 10
```

If successful:

```text id="u8a4dh"
→ Target Match Registered
```

Otherwise:

```text id="oz3e4u"
→ Escalate to SIFT Fallback
```

---

## 🔹 Tier 3 — SIFT Fallback Engine

High-precision fallback matcher using:

* Difference of Gaussian (DoG)
* Scale-space extrema detection
* 128D orientation descriptors
* FLANN KD-Tree nearest-neighbor search

---

## Lowe's Ratio Test

To eliminate ambiguous feature correspondences:

```text id="eq7xwy"
d1 / d2 < 0.7
```

Where:

| Variable | Description                      |
| -------- | -------------------------------- |
| d1       | Nearest-neighbor distance        |
| d2       | Second nearest-neighbor distance |

### Benefits

✅ Reduces false positives
✅ Improves structural consistency
✅ Preserves spatial reliability

---

# ⚡ Concurrency & Memory Optimization

# 🧵 Asynchronous Parallelism

Computationally intensive feature-matching operations are distributed across concurrent worker threads.

Implementation:

```python id="rx4k3t"
ThreadPoolExecutor
```

### Advantages

* Parallel ROI processing
* Lower processing latency
* Improved throughput scalability

---

# 💾 Deterministic Cache Layer

Inference responses are cached using SHA-256 hashing:

```text id="r2t0ba"
SHA-256(image_binary_payload)
```

Cached data includes:

* Detection coordinates
* Bounding boxes
* Spatial metadata
* Matching states

Serialization engine:

```python id="v4z9tm"
pickle
```

---

# 🔒 Thread-Safe Serialization

Concurrent disk operations are protected using:

```python id="n7e3wb"
threading.Lock()
```

Ensuring:

* Atomic writes
* Cache consistency
* Race-condition prevention

---

# 🛠 Technology Stack

| Category          | Technology         |
| ----------------- | ------------------ |
| Backend Framework | Flask              |
| Computer Vision   | OpenCV             |
| Object Detection  | Roboflow           |
| Concurrency       | concurrent.futures |
| Hashing           | hashlib            |
| Serialization     | pickle             |
| Synchronization   | threading.Lock     |
| Feature Matching  | ORB / SIFT         |

---

# 📈 Performance Strategy

The platform prioritizes computational efficiency through:

* Cascading validation filters
* Early candidate rejection
* Multi-threaded ROI processing
* Persistent inference caching
* Resolution-aware scaling
* Adaptive luminance normalization

---

# 🎯 Key Capabilities

✅ Real-time bookshelf tracking
✅ Deep-learning assisted localization
✅ Illumination-aware preprocessing
✅ High-dimensional feature matching
✅ Concurrent asynchronous execution
✅ Deterministic cache acceleration
✅ Structural object re-identification

---

# 📁 Project Structure

```text id="vh3s8u"
Bookshelf-Catalog/
│
├── app.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   ├── uploads/
│   └── results/
│
├── cache/
│   └── detection_cache.pkl
│
├── models/
│
└── utils/
    ├── matching.py
    ├── preprocessing.py
    └── cache_manager.py
```

---

# ⚡ Installation

# 1️⃣ Clone the Repository

```bash id="n1f0up"
git clone https://github.com/your-username/bookshelf-catalog.git

cd bookshelf-catalog
```

---

# 2️⃣ Create Virtual Environment

## Linux / macOS

```bash id="m8k5ys"
python3 -m venv venv

source venv/bin/activate
```

## Windows

```bash id="w7r0kh"
venv\Scripts\activate
```

---

# 3️⃣ Install Dependencies

```bash id="u9x6fa"
pip install -r requirements.txt
```

---

# 4️⃣ Configure Environment Variables

Create a `.env` file:

```env id="y2z9tw"
ROBOFLOW_API_KEY=your_roboflow_api_key
```

---

# 🚀 Running the Application

Start the Flask development server:

```bash id="p4f6nm"
python app.py
```

The application will be available at:

```text id="t0q7wb"
http://localhost:5000
```

---

# 🔮 Future Improvements

Planned enhancements include:

* Real-time webcam integration
* GPU-accelerated feature extraction
* YOLO-based local inference support
* Vector database indexing
* Multi-user dashboard system
* Book metadata OCR integration
* Dockerized deployment infrastructure
* Kubernetes orchestration support

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Developed as an advanced computer vision and object re-localization research project focused on hybrid AI-assisted bookshelf digitization systems.
