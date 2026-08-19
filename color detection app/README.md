# Industrial AI Color Vision & Real-Time Analytics Platform

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Pytest](https://img.shields.io/badge/Tests-8%20Passed-brightgreen.svg)](https://pytest.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-orange.svg)](https://opencv.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML%20K--Means-yellow.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industrial-grade Computer Vision and Machine Learning platform engineered for real-time spatial object segmentation, high-throughput colorimetry, perceptual color classification via **CIELAB ($\Delta E$) $k$-d Trees**, and unsupervised **K-Means Dominant Color Palette Extraction**.

Built for 4th-Year Computer Science & Engineering (AI/ML) Capstone Portfolios, Quality Inspection Automation, and Autonomous Robotic Vision.

---

## Table of Contents

- [Overview](#overview)
- [Key Technical Innovations](#key-technical-innovations)
- [Mathematical Formulation](#mathematical-formulation)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Usage & Execution Modes](#usage--execution-modes)
- [Performance & Latency Benchmarks](#performance--latency-benchmarks)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Academic & Technical References](#academic--technical-references)
- [License](#license)

---

## Overview

Traditional color detection systems rely on naive RGB Euclidean distance heuristics and monolithic scripts, which suffer from perceptual non-uniformity and poor computational scalability.

This platform resolves these limitations by introducing:
1. **Perceptually Uniform Color Matching**: Uses standard CIE $L^*a^*b^*$ colorimetry combined with multidimensional $k$-d tree spatial indexing to classify colors against an 865+ shade dataset in sub-35 microseconds.
2. **Unsupervised Machine Learning**: Applies Mini-Batch K-Means clustering to extract dominant color palettes, calculate cluster frequency proportions, and visualize segmented color distributions.
3. **Multi-Target Spatial Segmentation**: Leverages HSV thresholding, morphological opening/closing operations, contour hierarchy analysis, and centroid crosshair tracking.
4. **Thread-Safe Video Processing**: Decouples hardware frame acquisition from GUI rendering using asynchronous worker threads and real-time FPS monitoring.

---

## Key Technical Innovations

| Feature | Legacy Approach | Industrial AI Platform (Current) |
| :--- | :--- | :--- |
| **Color Distance Metric** | Naive Manhattan/Euclidean RGB ($L_1$/$L_2$) | **Perceptually Uniform CIELAB ($\Delta E_{76}$)** matching human eye sensitivity |
| **Nearest Neighbor Search** | $O(N)$ linear loop | **$O(\log N)$ Spatial $k$-d Tree Indexing** ($32.74\,\mu\text{s}$ latency, $>30,500$ queries/sec) |
| **Machine Learning** | None (Static rule-based) | **Unsupervised MiniBatch K-Means Clustering** for dominant palette extraction |
| **Video Processing** | Blocking main loop | **Thread-Safe Asynchronous Video Stream Engine** with real-time FPS telemetry |
| **User Interface** | Subprocess shell calls | **Unified CustomTkinter Dashboard** with embedded video canvas & WCAG contrast badges |
| **Software Architecture** | Flat procedural scripts | **Modular MVC Clean Architecture** (`src/core`, `src/gui`, `config`, `tests`) |

---

## Mathematical Formulation

### 1. Perceptual Color Difference ($\Delta E_{76}$)

Standard RGB color spaces are perceptually non-uniform: two pairs of colors with identical Euclidean distances in RGB coordinates are perceived with substantially different visual contrasts by the human visual cortex.

The system converts sRGB signals to CIE $L^*a^*b^*$ coordinates ($L^*$ = Perceptual Lightness, $a^*$ = Green-Red chromatic axis, $b^*$ = Blue-Yellow chromatic axis) and evaluates perceptual distance:

$$\Delta E_{ab}^* = \sqrt{(L_2^* - L_1^*)^2 + (a_2^* - a_1^*)^2 + (b_2^* - b_1^*)^2}$$

Interpretation scale:
- $\Delta E \le 1.0$: Imperceptible to the human eye.
- $1.0 < \Delta E \le 2.3$: Just Noticeable Difference (JND).
- $2.3 < \Delta E \le 10.0$: Perceptible at a glance.
- $\Delta E > 10.0$: Colors perceived as distinct categories.

### 2. K-Means Dominant Color Clustering

To extract dominant palettes from an image or region of interest (ROI) $\mathbf{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_N\} \subset \mathbb{R}^3$, the unsupervised K-Means algorithm partitions pixels into $K$ distinct clusters by minimizing the within-cluster sum of squares (WCSS):

$$\arg\min_{\mathbf{S}} \sum_{i=1}^{K} \sum_{\mathbf{x} \in S_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$

Where $\boldsymbol{\mu}_i$ represents the mean centroid of cluster $S_i$. Each cluster proportion is calculated as:

$$P(S_i) = \frac{|S_i|}{N} \times 100\%$$

---

## System Architecture

```mermaid
graph TD
    subgraph Ingestion Layer
        A[Webcam Feed / Camera Stream] --> VS[Async VideoStream Engine]
        B[Static Images / Files] --> PP[Image Preprocessor]
    end

    subgraph Core AI/ML Engine
        VS --> PP
        PP --> CC[ColorClassifier - CIELAB KD-Tree]
        PP --> KM[DominantColorExtractor - K-Means]
        PP --> OT[ColorObjectTracker - Spatial Segmentation]
        Dataset[(colors.csv 865+ Shades)] --> CC
    end

    subgraph Presentation & Control Layer
        CC --> App[ColorVisionApp GUI]
        KM --> App
        OT --> App
        App --> Analytics[Telemetry, CSV/JSON Log Exporter]
        CC --> CLI[CLI & Benchmark Engine]
    end
```

---

## Repository Structure

```
project/
├── config/
│   ├── __init__.py
│   └── settings.py               # Centralized Dataclass configurations
├── data/
│   ├── raw/
│   │   └── colors.csv            # 865+ Named Colors Reference Dataset
│   └── sample_images/
│       └── colorpic.jpg          # High-resolution benchmark test image
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── color_classifier.py   # CIELAB Delta E + KD-Tree Engine
│   │   ├── color_extractor.py    # Unsupervised K-Means Palette Extractor
│   │   ├── object_tracker.py     # Multi-Color Spatial Centroid Tracker
│   │   └── video_stream.py       # Asynchronous Video Capture Thread
│   ├── gui/
│   │   ├── __init__.py
│   │   └── app.py                # Unified Modern CustomTkinter Dashboard
│   └── utils/
│       ├── __init__.py
│       ├── image_processing.py   # WCAG contrast, morphology, resizing
│       └── logger.py             # Structured Logging System
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py        # Unit tests for KD-Tree & CIELAB
│   └── test_extractor.py         # Unit tests for K-Means clustering
├── main.py                       # Unified CLI and GUI Entrypoint
├── requirements.txt              # Production dependencies
├── LICENSE                       # MIT License
└── README.md                     # Technical Documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Webcam (optional, required for live video tracking mode)

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd project
   ```

2. **Create and activate a virtual environment (optional but recommended)**:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage & Execution Modes

### 1. Graphical User Interface (Default)
Launch the modern CustomTkinter dashboard:
```bash
python main.py
```
- **Tab 1 (Image Inspector & K-Means Palette):** Interactive click-to-identify color picking, cluster count slider ($K=2 \dots 8$), and proportional color swatch strips.
- **Tab 2 (Live Real-Time Object Tracker):** Embedded video stream with real-time tracking modes (All Colors, Red, Green, Blue, Yellow, Orange, Purple, Multi-Simultaneous), telemetry metrics, and snapshot capture.
- **Tab 3 (Dataset & Analytics):** Searchable 865+ color table and one-click CSV/JSON detection log export.

### 2. High-Throughput Latency Benchmark
Evaluate KD-Tree indexing latency against naive linear search across 10,000 queries:
```bash
python main.py --benchmark
```

### 3. CLI Dominant Color Palette Extraction
Extract top $K$ dominant colors with percentages directly in the terminal:
```bash
python main.py -i data/sample_images/colorpic.jpg -k 5
```

### 4. Standalone High-Performance OpenCV Video Tracker
Run without GUI overhead for maximum frame rate:
```bash
# Track all saturated colors simultaneously
python main.py --cli -m all

# Track specific colors (e.g., red, green, blue)
python main.py --cli -m red
```

---

## Performance & Latency Benchmarks

Tested on standard x86-64 hardware with 10,000 random RGB query points against the 865 reference color dataset:

| Metric | Naive RGB Linear Search | CIELAB $k$-d Tree (This Project) |
| :--- | :--- | :--- |
| **Average Query Latency** | $247.77\,\mu\text{s}$ | **$32.74\,\mu\text{s}$** |
| **Throughput** | $\sim 4,036\,\text{queries/s}$ | **$> 30,500\,\text{queries/s}$** |
| **Speedup Factor** | $1.0\times$ (Baseline) | **$7.6\times$ Acceleration** |
| **Perceptual Accuracy** | Low (RGB Non-uniform) | **High (CIE 1976 Standard)** |

---

## Testing & Quality Assurance

Run the automated test suite with `pytest`:

```bash
python -m pytest tests/ -v
```

### Test Suite Summary
- `tests/test_classifier.py::test_dataset_loaded` : PASSED
- `tests/test_classifier.py::test_classify_pure_red` : PASSED
- `tests/test_classifier.py::test_classify_pure_green` : PASSED
- `tests/test_classifier.py::test_classify_pure_blue` : PASSED
- `tests/test_classifier.py::test_batch_classification` : PASSED
- `tests/test_classifier.py::test_rgb_bounds_clamping` : PASSED
- `tests/test_extractor.py::test_palette_extraction_synthetic` : PASSED
- `tests/test_extractor.py::test_palette_k_bounds` : PASSED

---

## Academic & Technical References

1. CIE (1978). *Recommendations on Uniform Color Spaces, Color-Difference Equations, Psychometric Color Terms*. Supplement No. 2 to CIE Publication No. 15 (E-1.3.1) 1971.
2. Bentley, J. L. (1975). *Multidimensional binary search trees used for associative searching*. Communications of the ACM, 18(9), 509-517.
3. MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations*. Proc. 5th Berkeley Symp. Math. Statist. Prob., 281-297.
4. World Wide Web Consortium (W3C). *Web Content Accessibility Guidelines (WCAG) 2.0 - Contrast Ratio Standard*. W3C Recommendation.

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
