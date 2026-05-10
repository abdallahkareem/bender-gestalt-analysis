<div align="center">

# 🧠 BGT-Assessment
### Automated Vision-Based Scoring of the Bender Gestalt Test

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Academic%20Project-orange?style=for-the-badge)]()

> An AI system that **automatically analyzes hand-drawn BGT figures** and detects neuropsychological scoring errors — replacing manual clinical assessment with computer vision and geometric analysis.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Pipeline](#-pipeline)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Error Detection](#-error-detection)
- [Scoring System](#-scoring-system)
- [Results](#-results)
- [Team](#-team)

---

## 🔬 Overview

The **Bender Gestalt Test (BGT)** is a neuropsychological tool where patients copy 9 geometric figures. Clinicians then manually identify errors such as rotation, distortion, and perseveration to detect organic brain damage or cognitive regression.

This project automates that scoring using a full **computer vision pipeline**:

| Manual Process | Our System |
|---------------|------------|
| Clinician examines drawings by eye | OpenCV preprocessing pipeline |
| Subjective error detection | Geometric metrics (Hausdorff, Hu Moments) |
| No consistency guarantee | Reproducible, quantifiable scores |
| Minutes per patient | Sub-second analysis |

---

## 🔄 Pipeline

```
Raw Drawing (scan/photo)
        │
        ▼
┌─────────────────────┐
│   PREPROCESSING     │  Gaussian Blur → NLM Denoise → CLAHE → Otsu Binarization → Skeletonization
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  SHAPE COMPARISON   │  Hu Moments · Hausdorff Distance · Rotation Angle · Contour Match
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  ERROR DETECTION    │  Rotation · Distortion · Perseveration · Integration Failure · Fragmentation
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  CLINICAL SCORING   │  Koppitz System (0–18) → Impairment Classification
└─────────────────────┘
        │
        ▼
  Report (JSON + HTML + Annotated Images)
```

---

## 📁 Project Structure

```
BGT-Assessment/
│
├── 📄 main.py                        # Entry point — single or batch processing
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 📂 src/
│   ├── 📂 preprocessing/
│   │   └── pipeline.py               # 5-stage image preprocessing
│   │
│   ├── 📂 algorithm/
│   │   ├── comparator.py             # Template matching & geometric metrics
│   │   └── detector.py               # Clinical error classification
│   │
│   ├── 📂 evaluation/
│   │   └── scorer.py                 # Koppitz scoring → impairment level
│   │
│   └── 📂 utils/
│       ├── visualizer.py             # HTML reports + annotated images
│       └── logger.py                 # Centralized logging
│
├── 📂 data/
│   ├── raw_drawings/                 # 89 patient drawing scans (not tracked by git)
│   ├── templates/                    # 9 canonical BGT reference figures
│   └── processed/                    # Pipeline output cache
│
├── 📂 notebooks/
│   ├── 01_preprocessing_demo.ipynb   # Visual walkthrough of each stage
│   ├── 02_metric_analysis.ipynb      # Hu Moments & Hausdorff exploration
│   └── 03_scoring_evaluation.ipynb   # Score distribution & validation
│
└── 📂 tests/
    ├── test_preprocessing.py
    ├── test_comparator.py
    └── test_scorer.py
```

---

## 🗄️ Dataset

| Split | Images | Description |
|-------|--------|-------------|
| Patient Drawings | **89** | Real hand-drawn BGT scans (varying paper, pencil pressure) |
| Reference Templates | **9** | Canonical BGT figures (A, 1–8) used for comparison |

**Data is not included in this repo** (patient privacy). Place your drawings in `data/raw_drawings/` and templates in `data/templates/`.

Template naming convention:
```
data/templates/templateA.png
data/templates/template1.png
...
data/templates/template8.png
```

---

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/BGT-Assessment.git
cd BGT-Assessment

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Single Image
```bash
python main.py --input data/raw_drawings/patient_01.png --visualize
```

### Batch Processing (all 89 images)
```bash
python main.py --input data/raw_drawings/ --batch --visualize --output results/
```

### Output
```
results/
├── patient_01_preprocessing.png   # 6-stage preprocessing grid
├── patient_01_errors.png          # Error overlay on original
├── patient_01_report.html         # Full clinical HTML report
└── batch_results.json             # Machine-readable scores for all patients
```

---

## 🔍 Error Detection

The system detects **5 clinically defined BGT errors**:

| Error | Detection Method | Clinical Meaning |
|-------|-----------------|------------------|
| **Rotation** | Principal axis angle deviation ≥ 45° | Motor / spatial planning deficit |
| **Distortion** | Hu similarity < 0.40 + Hausdorff > 60px | Perceptual-motor dysfunction |
| **Perseveration** | Hough circle count vs. expected | Inability to shift cognitive set |
| **Integration Failure** | Connected component count > expected | Figure-ground / synthesis difficulty |
| **Fragmentation** | Broken contour segments ≥ 3 | Motor control / tremor indicators |

---

## 📊 Scoring System

Based on the **Koppitz (1963)** scoring system:

```
Score   Impairment Level
─────────────────────────
  0     No Impairment
 1–4    Minimal Impairment
 5–8    Mild Impairment
 9–12   Moderate Impairment
 13–18  Severe Impairment
```

Each of the 9 figures receives 0–2 error points. Maximum total: **18 points**.

---

## 📈 Results

Sample output on a single drawing:

```json
{
  "total_score": 6,
  "max_score": 18,
  "impairment_level": "mild_impairment",
  "summary": {
    "figures_with_errors": 4,
    "most_affected_figures": ["4", "7", "A"],
    "most_common_error": "rotation"
  }
}
```

---

## 👥 Team

| Name | ID |
|------|----|
| Member 1 | 2023XXX |
| Member 2 | 2023XXX |
| Member 3 | 2023XXX |
| Member 4 | 2023XXX |
| Member 5 | 2023XXX |

*Computing Cognitive Course — Spring 2026*

---

<div align="center">
  <sub>Built with OpenCV · scikit-image · NumPy · SciPy</sub>
</div>
