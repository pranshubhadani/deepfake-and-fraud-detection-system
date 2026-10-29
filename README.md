# Deepfake & Fraud Detection System

> **2nd Year 3rd Sem APP Project**  
> A Unified Multi-Modal Defense System against Deepfake Videos, Scam Links, and Fraudulent SMS Messages.

---

## 🌟 System Architecture Overview

```
                        ┌────────────────────────────────────────┐
                        │       Multi-Modal Incident Input       │
                        │  (SMS Text, Phishing URL, Video/Photo) │
                        └───────────────────┬────────────────────┘
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────┐
         │                                  │                                  │
         ▼                                  ▼                                  ▼
┌──────────────────┐               ┌──────────────────┐               ┌──────────────────┐
│  link_detector   │               │   sms_detector   │               │deepfake_detector │
│ (Heuristics &    │               │  (TF-IDF Vector+ │               │ (OpenCV Keyframes│
│  Domain Age)     │               │   Naive Bayes)   │               │ + PyTorch CNN)   │
└────────┬─────────┘               └────────┬─────────┘               └────────┬─────────┘
         │                                  │                                  │
   Link_Suspicion                     SMS_Spam_Prob                  Deepfake_Artifacts
     (Weight 0.4)                      (Weight 0.3)                     (Weight 0.3)
         │                                  │                                  │
         └──────────────────────────────────┼──────────────────────────────────┘
                                            │
                                            ▼
                        ┌────────────────────────────────────────┐
                        │      hybrid_detector.py Evaluator      │
                        │  Final_Score = (Link*0.4) + (SMS*0.3)  │
                        │             + (Deepfake*0.3)           │
                        └───────────────────┬────────────────────┘
                                            │
                                            ▼
                               [ Comprehensive Threat Report ]
```

---

## 🧩 Modules Breakdown

### 1. `deepfake_detector.py` (Deepfake Video & Image Analysis)
- **OpenCV Keyframe Sampling:** Samples representative video frames at regular intervals (every 15th frame) to conserve compute resources.
- **Face Alignment & Bounding Cropper:** Locates facial landmarks with Haar Cascades and crops with extra boundary padding (25%) to preserve hairline and jawline blending seams.
- **Visual Boundary & Artifact CNN:** 4-stage convolutional neural network (or ResNet18 transfer learning) trained to detect warping, unnatural lighting, and frequency inconsistencies.

### 2. `sms_detector.py` (SMS Scam & Phishing Text Detection)
- Preprocesses messages and trains a **TF-IDF + Multinomial Naive Bayes** pipeline on SMS datasets to detect financial fraud, lottery scams, and urgent social engineering.

### 3. `link_detector.py` (Fraud Link Verification)
- Inspects URLs for suspicious features: missing HTTPS, excessive length, obfuscated symbols (`@`, excessive hyphens), domain age via WHOIS/RDAP, and hidden forms.

### 4. `hybrid_detector.py` (Unified Multi-Modal Incident Scoring)
- Integrates all three subsystems using the **Hybrid Verdict Formula**:
  $$\text{Final\_Score} = (\text{Link\_Suspicion} \times 0.4) + (\text{SMS\_Spam\_Prob} \times 0.3) + (\text{Deepfake\_CNN\_Artifacts} \times 0.3)$$

---

## 📁 Dataset Organization & Suggested Sizes

Organize your deepfake images/videos in `Datasets/Deepfake Datasets/`:

```
Datasets/
└── Deepfake Datasets/
    ├── train/
    │   ├── real/          # Authentic face images (.jpg, .png)
    │   └── fake/          # Deepfake face images (.jpg, .png)
    ├── val/
    │   ├── real/          # Validation authentic images
    │   └── fake/          # Validation deepfake images
    └── test_videos/       # MP4 / AVI video files for testing
```

### 📊 Dataset Size Recommendations:

| Goal / Setup | Recommended Size | Balanced Ratio | Approx. Training Time | Storage |
| :--- | :--- | :--- | :--- | :--- |
| **Quick Prototype / CPU** | **2,000 – 4,000 images** | 50% Real / 50% Fake | ~10 – 15 mins (CPU) | ~150 MB |
| **College Project (Recommended)** | **10,000 – 20,000 images** | 5,000 Real / 5,000 Fake | ~30 – 60 mins (GPU/Colab) | ~1.2 GB |
| **High Accuracy / Benchmark** | **40,000 – 100,000 images** | Balanced | 2 – 4 hours (CUDA GPU) | ~5 GB+ |

### 🌐 Recommended Public Datasets:
1. **Kaggle 140k Real and Fake Faces** *(Easiest to start: already pre-cropped into real/fake folders)*
2. **Celeb-DF (v2)** *(High visual quality real and deepfake video benchmark)*
3. **FaceForensics++ (FF++)** *(Standard benchmark covering Deepfakes, Face2Face, FaceSwap)*

---

## 🚀 How to Run

### 🌐 Launch the Interactive Web Dashboard (Recommended):
```bash
python -m streamlit run app.py
```
> Opens the full multi-modal cybersecurity dashboard in your browser at `http://localhost:8501`.

### Run Deepfake Detection (Interactive Video/Image CLI Scanner & Trainer):
```bash
python deepfake_detector.py
```

### Run Full Multi-Modal Hybrid Threat Scanner:
```bash
python hybrid_detector.py
```

### Run Individual Text or Link Detectors:
```bash
python sms_detector.py
python link_detector.py
```
