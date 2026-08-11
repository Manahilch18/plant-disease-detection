<div align="center">

# 🌿 Plant Disease Detection System

### End-to-End Deep Learning Application for Automated Plant Disease Classification

*Custom CNN classification · CLIP-based validity gating · Grad-CAM explainability · FastAPI inference · Streamlit UI · Docker deployment*

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Transformers-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Inference%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#-license)

</div>


## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [Exploratory Data Analysis](#-exploratory-data-analysis-eda)
- [Feature Engineering](#️-feature-engineering)
- [Data Preprocessing](#-data-preprocessing)
- [Model Training](#-model-training)
- [Machine Learning Algorithms Used](#-machine-learning-algorithms-used)
- [Model Comparison](#-model-comparison)
- [Best Model](#-best-model)
- [Performance Metrics](#-performance-metrics)
- [Model Explainability — Grad-CAM](#-model-explainability--grad-cam)
- [Key Insights](#-key-insights)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Docker Installation](#-docker-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Results](#-results)
- [Deployment Architecture](#-deployment-architecture)
- [Future Improvements](#-future-improvements)
- [Reproducibility](#-reproducibility)
- [Author](#-author)
- [License](#-license)


## 📌 Project Overview

Plant diseases significantly reduce crop yield and threaten agricultural productivity worldwide. Early, accurate, and scalable disease identification can help farmers and agronomists intervene before a disease spreads across a field.

This project delivers a **production-style, end-to-end plant disease detection system** that goes beyond a single trained model — it is a complete AI application covering data preparation, model comparison and selection, explainability, input validation, and containerized deployment.

The system:

* 🖼️ Accepts a plant leaf image from the user
* 🔍 Validates that the image is actually a plant leaf **before** running inference — using a CLIP-based zero-shot gate, not a confidence threshold
* 🧠 Classifies the leaf into one of **38 disease/healthy classes** using a custom-trained CNN
* 📊 Returns the predicted class and a calibrated confidence score
* 🔥 Generates a **Grad-CAM** heatmap explaining which regions of the leaf influenced the prediction
* 📜 Persists a searchable, filterable **prediction history** for auditability
* 🚀 Exposes inference through a **FastAPI** REST service
* 🎨 Provides an interactive, dark-themed **Streamlit** front end
* 🐳 Ships as two **Docker**-orchestrated services for reproducible deployment

### System Workflow

```text
User Uploads Leaf Image
        │
        ▼
┌──────────────────────────┐
│     Streamlit Web App    │
└────────────┬──────────────┘
             │
             ▼
┌──────────────────────────┐
│   Image Validity Gate    │
│   (CLIP zero-shot check) │
└────────────┬──────────────┘
             │  valid leaf?
     ┌───────┴────────┐
     │no               │yes
     ▼                 ▼
 Reject / Warn   ┌──────────────────────┐
                 │      FastAPI API      │
                 └────────────┬───────────┘
                              ▼
                 ┌──────────────────────┐
                 │   Baseline CNN Model  │
                 │   38-class classifier │
                 └────────────┬───────────┘
                        ┌─────┴─────┐
                        ▼           ▼
                  Prediction     Grad-CAM
                        │           │
                        └─────┬─────┘
                              ▼
                   Results + Explanation
                              │
                              ▼
                   Saved to Prediction History
```

> 💡 **Engineering note:** The validity gate is a fully independent CLIP model that runs *before* the CNN and never inspects the CNN's own output. This was a deliberate design choice — a closed-set softmax classifier can be confidently wrong on out-of-domain inputs, so gating on the CNN's own confidence would not have solved the problem it was meant to solve.

## 📸 Demo / App Interface

| Analyze | Prediction Result | Model Comparison | Prediction History|
|---|---|---|---|
|(<img width="1802" height="505" alt="image" src="https://github.com/user-attachments/assets/b42fd2ba-1d47-4e0a-9e68-6758a867ddef" />) | (<img width="1884" height="920" alt="image" src="https://github.com/user-attachments/assets/f80e008a-a1cf-40e6-a4f8-299213d2cdb5" />) | (<img width="1774" height="908" alt="image" src="https://github.com/user-attachments/assets/c5146e75-babc-4b95-a056-5f02cdb3c2ae" />) | (<img width="1872" height="897" alt="image" src="https://github.com/user-attachments/assets/b20738fd-e82c-46a5-80cd-500b7af9ac20" />) |

## 🎯 Problem Statement

Traditional plant disease identification depends on manual inspection by farmers, agricultural extension workers, or plant pathologists. This process is:

* ⏱️ Time-consuming and doesn't scale to large farms
* 💰 Expensive to staff at scale
* 👤 Dependent on expert availability, especially in rural regions
* 🔁 Difficult to perform *consistently* across inspectors
* 🔬 Challenging for early-stage, visually subtle disease detection

**Objective:** Build a computer vision system that automatically classifies plant leaf images into their correct disease category, while also addressing a common blind spot in applied ML systems — **model interpretability and input validity** — rather than shipping a black-box label-only classifier.

---

## 📊 Dataset Description

The project uses the **PlantVillage Dataset**, a widely used benchmark dataset for plant disease classification research.

| Property | Value |
|---|---:|
| Dataset | PlantVillage |
| Total Images | 54,305 |
| Number of Classes | 38 |
| Image Size | 224 × 224 |
| Training Images | 38,013 |
| Validation Images | 8,146 |
| Test Images | 8,146 |
| Split Ratio | 70 / 15 / 15 |
| Split Strategy | Stratified |
| Task | Multi-class image classification |

### Dataset Organization

```text
plantvillage_resplit/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

> **Note:** The dataset is not included in this repository due to its size and distribution/licensing considerations. Refer to the official PlantVillage dataset source to obtain it.

---

## 🔎 Exploratory Data Analysis (EDA)

EDA was performed prior to model development to understand dataset structure, quality, and potential pitfalls.

**EDA covered:**

* Dataset size and cardinality analysis
* Per-class distribution analysis (identifying potential class imbalance)
* Image dimension and format inspection
* Sample image visualization across classes
* Class-label verification against the source taxonomy
* Train / validation / test distribution consistency checks
* Dataset metadata validation

### Dataset Validation Checklist

- [x] Expected number of classes (38) confirmed
- [x] Correct class-to-index mapping verified
- [x] Correct image dimensions confirmed (224 × 224)
- [x] Correct dataset cardinality across splits
- [x] Consistent train/validation/test metadata

---

## 🛠️ Feature Engineering

Unlike traditional tabular ML pipelines, this project uses **raw image pixels as the primary input representation** — there is no manual feature-engineering step.

Feature extraction is performed automatically by the convolutional layers of the CNN, which learn hierarchical visual representations directly from training data, including:

* Leaf edges and contours
* Surface texture
* Color patterns and discoloration
* Spot and lesion morphology
* Disease-specific structural patterns
* Overall leaf shape characteristics


## 🧹 Data Preprocessing

### Preprocessing Pipeline

```text
Raw Image
    ↓
Image Loading
    ↓
RGB Conversion
    ↓
Resize to 224 × 224
    ↓
Float32 Conversion
    ↓
Model Input
```

> ⚠️ **Critical implementation detail:** The Baseline CNN was trained on image pixel values in the **[0, 255]** range — **not** normalized to [0, 1]. Every stage of the pipeline (training, FastAPI inference, and the Grad-CAM explainability path) preserves this exact representation. This constraint is enforced end-to-end and is explicitly called out in code comments to prevent an accidental `/255` regression from silently degrading model accuracy.

### TensorFlow Data Pipeline

```python
tf.data.Dataset
    → cache()
    → prefetch(tf.data.AUTOTUNE)
```

Caching and prefetching reduce input-pipeline bottlenecks during both training and inference.


## 🧠 Model Training

The primary classifier is a custom **Convolutional Neural Network (CNN)** built with the TensorFlow/Keras Functional API.

### Baseline CNN Architecture

```text
Input (224×224×3)
 ↓
[Conv2D → BatchNorm → ReLU → MaxPooling]  × 5 blocks
 ↓
Global Average Pooling
 ↓
Dropout
 ↓
Dense
 ↓
Dropout
 ↓
Dense (38-class Softmax)
```

### Training Configuration

| Parameter | Value |
|---|---:|
| Image Size | 224 × 224 |
| Number of Classes | 38 |
| Batch Size | 32 |
| Epochs | 30 |
| Learning Rate | 0.001 |
| Dropout | 0.3 |
| L2 Weight Decay | 0.0001 |
| Random Seed | 42 |
| Framework | TensorFlow / Keras |

The best-performing checkpoint (selected on validation performance) is stored at:

```text
artifacts/baseline_cnn/models/best_model.keras
```

---

## 🤖 Machine Learning Algorithms Used

### 1. Convolutional Neural Network — Primary Classifier
A custom CNN serves as the core disease classifier. CNNs are well-suited to this task because convolutional filters learn spatial, translation-invariant patterns directly from raw pixels, without manual feature design.

### 2. Transfer Learning Baselines — EfficientNetB0 & ResNet50
Two ImageNet-pretrained architectures were evaluated as alternative candidates, to validate whether transfer learning would outperform a task-specific custom CNN on this dataset (see [Model Comparison](#-model-comparison)).

### 3. CLIP-Based Image Validity Gate
A pretrained CLIP model (`openai/clip-vit-base-patch32`) provides a zero-shot check on whether an uploaded image plausibly depicts a plant leaf, **before** it reaches the CNN.

```text
Uploaded Image
      ↓
CLIP Zero-Shot Similarity Check
 (vs. "a plant leaf" and non-leaf prompts)
      ↓
   Plant Leaf?
   /        \
 Yes         No
  ↓           ↓
 CNN      Reject / Warn
Prediction  (no CNN call, no history write)
```

> **Why not just threshold the CNN's confidence?** A closed-set softmax classifier always outputs a class, even for out-of-domain images (a car, a person, a random object) — and it can do so *confidently*. Confidence alone cannot distinguish "the model is sure this is Apple Scab" from "the model was never taught this isn't a leaf at all." The CLIP gate solves a fundamentally different problem — domain membership — independently of the CNN's own output.

### 4. Grad-CAM — Explainability, Not Classification
Grad-CAM (Gradient-weighted Class Activation Mapping) is used purely for **post-hoc interpretability** — it does not participate in the classification decision.

---

## 📈 Model Comparison

Three architectures were trained and evaluated under identical test conditions on the PlantVillage test set (**8,146 images, 38 classes**):

| Metric | Baseline CNN | EfficientNetB0 | ResNet50 |
|---|---:|---:|---:|
| **Test Loss** | 0.1589 | 0.0687 | 1.5797 |
| **Accuracy** | **98.85%** | 97.70% | 65.52% |
| **Top-5 Accuracy** | **99.96%** | **99.96%** | 90.69% |
| **Precision** | **98.25%** | 97.45% | 73.88% |
| **Recall** | **98.69%** | 96.59% | 65.52% |
| **Macro F1** | **98.45%** | 96.96% | 58.06% |
| **Weighted F1** | **98.85%** | 97.70% | 64.35% |
| **Evaluation Time** | 28.60s | **18.25s** | 34.13s |
| **Mean Confidence** | 98.58% | 98.01% | 81.18% |
| **Median Confidence** | 99.99% | **100.00%** | 91.00% |
| **Training** | 30 epochs | 25 actual / 30 planned | 45 epochs |

*(Bold indicates the best result for that row.)*

### 🧪 Training Strategy Summary

| Model | Training Approach | Epochs |
|---|---|---:|
| Baseline CNN | Custom CNN trained from scratch | 30 |
| EfficientNetB0 | Frozen ImageNet backbone (Phase 1) → fine-tuning (Phase 2) | 25 actual / 30 planned |
| ResNet50 | Transfer learning + fine-tuning | 45 |

> EfficientNetB0 training was planned for 30 epochs; **EarlyStopping halted training at epoch 25**. The best fine-tuning checkpoint occurred at **epoch 20**, reaching **97.85% validation accuracy** and **0.0639 validation loss**.

### Model-by-Model Analysis

<details>
<summary><strong>🧠 Baseline CNN</strong></summary>

- Strongest overall test accuracy (98.85%) among all evaluated models
- Excellent Top-5 accuracy (99.96%)
- Highest Macro F1 (98.45%) and Weighted F1 (98.85%) — indicating balanced performance across all 38 classes, not just the majority classes
- Highest recall (98.69%), minimizing missed disease detections
- **Selected as the final production model**

</details>

<details>
<summary><strong>⚡ EfficientNetB0</strong></summary>

- Strong overall classification performance and the fastest evaluation time among the three (18.25s)
- Benefited from ImageNet-pretrained representations via transfer learning
- Achieved slightly lower classification metrics in this evaluation, although it had a shorter evaluation time
- Therefore not selected as the final model for this project — a legitimate, competitive alternative rather than a failed candidate

</details>

<details>
<summary><strong>🔬 ResNet50</strong></summary>

- Substantially lower test accuracy (65.52%) and Macro F1 (58.06%) than the other two models
- Lower Top-5 accuracy (90.69%) relative to Baseline CNN and EfficientNetB0
- Achieved substantially lower performance under the configuration evaluated in this project
- Therefore not selected as the final model

</details>

---

## 🏆 Best Model

### Baseline CNN — `best_model.keras`

**Baseline CNN was selected as the final model based on its strongest overall classification performance across the evaluated models**, as measured on the PlantVillage held-out test set.

| Metric | Score |
|---|---:|
| 🥇 Test Accuracy | **98.85%** |
| 🥇 Top-5 Accuracy | **99.96%** |
| 🥇 Macro F1 | **98.45%** |
| 🥇 Weighted F1 | **98.85%** |
| 🥇 Recall | **98.69%** |
| Mean Confidence | 98.58% |

**Why the Baseline CNN, and not a transfer-learning model?**

* Achieved the highest accuracy, recall, Macro F1, and Weighted F1 of all three evaluated architectures
* Matched EfficientNetB0's excellent Top-5 accuracy (99.96%)
* Maintained very high, well-calibrated confidence scores across predictions
* Purpose-built for this exact classification task, rather than adapted from an unrelated domain
* Fully compatible with the project's Grad-CAM explainability pipeline
* Lightweight enough for efficient packaging and inference in a containerized FastAPI service

> **Model selection was based on comparative evaluation across multiple metrics rather than a single number** — accuracy, Top-5 accuracy, precision, recall, Macro/Weighted F1, evaluation time, and confidence calibration were all considered jointly. EfficientNetB0 remains a credible alternative, particularly where evaluation latency is the binding constraint; **it is not characterized as inferior in absolute terms, only as achieving slightly lower classification metrics in this specific evaluation.**

---

## 📏 Performance Metrics

Final Baseline CNN performance on the **8,146-image PlantVillage test set**:

| Metric | Result |
|---|---:|
| Accuracy | **98.85%** |
| Precision | **98.25%** |
| Recall | **98.69%** |
| Macro F1-Score | **98.45%** |
| Weighted F1-Score | **98.85%** |
| Top-5 Accuracy | **99.96%** |
| Test Loss | 0.1589 |
| Mean Confidence | 98.58% |
| Median Confidence | 99.99% |
| Evaluation Time | 28.60s |
| Test Samples | 8,146 |

### Metric Definitions

| Metric | What it measures |
|---|---|
| **Accuracy** | Overall proportion of correctly classified images across all 38 classes |
| **Precision** | Of the images predicted as a given disease, how many actually belong to it |
| **Recall** | Of the actual cases of a given disease, how many were correctly identified |
| **Macro F1** | Unweighted average F1 across all 38 classes — sensitive to performance on minority classes |
| **Weighted F1** | F1 averaged by class support — reflects performance weighted by real class frequency |
| **Top-5 Accuracy** | Whether the correct class appears among the model's top 5 predictions |

> **Why report Macro *and* Weighted F1?** In a 38-class agricultural dataset, class distributions are rarely perfectly balanced. Macro F1 exposes whether the model is quietly underperforming on rarer disease classes, even when overall accuracy looks strong — Weighted F1 shows performance as experienced across the actual class distribution. Reporting only accuracy would have hidden this distinction.

---

## 🔥 Model Explainability — Grad-CAM

Deep learning models can achieve strong predictive performance while remaining difficult to interpret — a serious limitation in an agricultural decision-support context, where a user needs to trust *why* a diagnosis was made.

This project integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visually explain each prediction.

```text
Input Leaf Image
       ↓
CNN Prediction
       ↓
Predicted Disease Class
       ↓
Gradient Calculation (w.r.t. final conv layer)
       ↓
Activation Weighting
       ↓
Grad-CAM Heatmap
       ↓
Overlay on Original Image
```

The resulting heatmap helps answer:

> **"Which regions of the leaf influenced the model's decision?"**

**Scientific framing matters here:** Grad-CAM visualizes the image regions that contributed to the model's prediction — it is explicitly **not** presented as a precise disease-location detector. Warmer regions indicate stronger model attention, not confirmed lesion boundaries. This distinction is preserved directly in the UI copy to avoid overstating what the visualization proves.

---

## 💡 Key Insights

**1. Input validity is a prerequisite for trustworthy predictions, not an afterthought.**
A 38-class closed-set classifier will always output *some* class, even for a photo of a car or a person — and can do so with high confidence. Rather than relying on the CNN's own confidence score (which cannot detect out-of-domain inputs by construction), a separate CLIP-based zero-shot gate validates the image *before* it reaches the classifier.

**2. CNNs learn disease-relevant visual features automatically.**
No manual feature engineering was required — convolutional layers learned edge, texture, color, and lesion-pattern representations directly from labeled training images.

**3. A custom, purpose-built CNN outperformed larger pretrained architectures on this task.**
Both EfficientNetB0 and ResNet50 bring strong ImageNet-pretrained representations, but the task-specific Baseline CNN achieved the best accuracy, recall, and F1 scores in this evaluation — a reminder that transfer learning is not automatically superior to a well-designed custom architecture for a well-defined, sufficiently large dataset.

**4. Confidence alone does not explain a prediction.**
Grad-CAM adds a visual layer of interpretation on top of the confidence score, while being carefully framed as an attention visualization rather than a diagnostic tool.

**5. Training/inference preprocessing consistency is a correctness-critical detail.**
The Baseline CNN was trained on **[0, 255]**-range pixel values. This exact representation is preserved through the FastAPI inference path and the Grad-CAM pipeline — a silent `/255` normalization mismatch between training and inference would silently degrade real-world accuracy without raising any error.

**6. Separating the UI from the inference API simplifies deployment and scaling.**
The Streamlit front end and the FastAPI inference service are independently deployable Docker containers, communicating over HTTP — the model can be scaled, versioned, or replaced without redeploying the UI, and vice versa.

**7. Auditability matters in an applied ML product.**
Every successful prediction is persisted to a local, structured prediction history (disease, confidence, filename, timestamp), giving users a searchable record of past analyses without adding external database dependencies.

---

## 🧰 Technologies Used

**Programming**
- Python 3.12

**Machine Learning / Deep Learning**
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- CLIP (`openai/clip-vit-base-patch32`)
- NumPy
- OpenCV
- Pillow

**Data & Visualization**
- Pandas
- Matplotlib
- Plotly

**Backend**
- FastAPI
- Uvicorn

**Frontend**
- Streamlit

**Explainable AI**
- Grad-CAM

**Deployment**
- Docker
- Docker Compose

**Development Environment**
- Jupyter / Kaggle
- VS Code
- WSL 2
- Git / GitHub

---

## 📁 Project Structure

```text
plant-disease-detection/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── streamlit_app/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   └── components/
│       ├── gradcam.py
│       ├── model_comparison.py
│       └── prediction_history.py
│
├── src/
│   ├── disease_info.py
│   ├── validity_check.py
│   └── explainability/
│       └── gradcam.py
│
├── data/
│   └── metadata/
│       └── prediction_history.json
│
├── artifacts/
│   ├── baseline_cnn/
│   │   └── models/
│   │       └── best_model.keras
│   │
│   └── preprocessing/
│       └── class_mapping.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
│
├── requirements.txt
├── requirements_streamlit.txt
│
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── .dockerignore
├── .gitignore
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

**Windows**
```bash
.venv\Scripts\activate
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

For the API service:
```bash
pip install -r requirements.txt
```

For the Streamlit application:
```bash
pip install -r requirements_streamlit.txt
```

> **Note:** The CLIP validity gate requires `transformers` and `torch`. On first run, it downloads the `openai/clip-vit-base-patch32` weights (~605MB) from Hugging Face and caches them locally — subsequent runs load from cache with no further downloads.

---

## 🐳 Docker Installation

Docker is the recommended deployment method, since the project runs as two coordinated services.

**Build and start both services:**
```bash
docker compose up --build
```

Once running:

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |

**Run in detached mode:**
```bash
docker compose up -d
```

**Stop the services:**
```bash
docker compose down
```

**View running containers:**
```bash
docker ps
```

---

## 🚀 Usage

### Option 1 — Docker Compose (Recommended)

```bash
docker compose up
```

Open the Streamlit interface at `http://localhost:8501`, upload a plant leaf image, and the application will:

1. Validate the image via the CLIP-based leaf gate
2. Send the validated image to the FastAPI inference service
3. Run disease classification via the Baseline CNN
4. Display the predicted class and confidence score
5. Generate and display a Grad-CAM explanation
6. Save the result to the prediction history

### Option 2 — Run FastAPI Locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive API documentation: `http://localhost:8000/docs`

### Option 3 — Run Streamlit Locally

```bash
streamlit run streamlit_app/app.py
```

Then open `http://localhost:8501`.

> **Important:** When running Streamlit outside Docker, point the API URL at `localhost` rather than the Docker service name `api`:
> ```text
> # Local
> API_BASE_URL=http://localhost:8000
>
> # Inside Docker Compose
> API_BASE_URL=http://api:8000
> ```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict/` | Runs disease classification on an uploaded image |
| `POST` | `/gradcam/` | Generates a Grad-CAM visualization for an uploaded image |

Full request/response schemas are available via FastAPI's interactive docs at `http://localhost:8000/docs`.

---

## 📊 Results

```text
┌────────────────────────────────────────┐
│         Plant Disease Detection         │
├────────────────────────────────────────┤
│                                          │
│           Upload Leaf Image             │
│                   ↓                     │
│          Image Validity Check           │
│                   ↓                     │
│           Disease Prediction            │
│                   ↓                     │
│            Confidence Score             │
│                   ↓                     │
│          Grad-CAM Explanation           │
│                   ↓                     │
│        Saved to Prediction History      │
│                                          │
└────────────────────────────────────────┘
```

### Example Output

```text
Predicted Disease:   Apple___Apple_scab
Confidence:          94.20%
Confidence Status:   High Confidence
Explainability:      Grad-CAM heatmap generated successfully
```

### Final Test-Set Evaluation (Baseline CNN)

```text
Test Accuracy:       98.85%
Top-5 Accuracy:      99.96%
Macro Precision:     98.25%
Macro Recall:        98.69%
Macro F1-Score:      98.45%
Weighted F1-Score:   98.85%
Test Loss:           0.1589
Test Samples:        8,146
```

---

## 🌐 Deployment Architecture

```text
                    Internet / User
                           │
                           ▼
                ┌────────────────────┐
                │    Streamlit UI     │
                │     Port 8501       │
                └─────────┬────────────┘
                          │  HTTP
                          ▼
                ┌────────────────────┐
                │     FastAPI API     │
                │     Port 8000       │
                └─────────┬────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   Baseline CNN      │
                │  best_model.keras   │
                └────────────────────┘
```

Docker Compose manages networking and startup ordering between the two services, so the UI can reliably reach the API by its service name (`api`) inside the Docker network.

---

## 🔮 Future Improvements

### Model
- [ ] Systematic hyperparameter optimization for the Baseline CNN
- [ ] Data augmentation strategies for underrepresented classes
- [ ] Formal class-imbalance mitigation (class weighting, focal loss)
- [ ] Structured per-class error analysis on misclassified test samples
- [ ] Evaluate additional architectures (MobileNet, Vision Transformers)

### Explainability
- [ ] Compare Grad-CAM against Grad-CAM++ and Score-CAM
- [ ] Quantitative evaluation of explanation faithfulness
- [ ] Additional XAI techniques (SHAP, Integrated Gradients) for cross-validation of explanations

### Application
- [x] Prediction history with search, filtering, and clear-history controls
- [ ] Batch image prediction
- [ ] Downloadable PDF/CSV prediction reports
- [ ] Expanded, richer disease-information content
- [ ] Multilingual support for non-English-speaking users

### Deployment
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] HTTPS termination
- [ ] CI/CD via GitHub Actions
- [ ] Automated container builds and image scanning
- [ ] Monitoring, logging, and alerting
- [ ] Production-grade API authentication
- [ ] Docker image size optimization

---

## 🧪 Reproducibility

To reproduce this project end-to-end:

1. Obtain the PlantVillage dataset
2. Organize it according to the expected directory structure
3. Configure the project environment (`requirements.txt` / `requirements_streamlit.txt`)
4. Run preprocessing and dataset validation
5. Train the Baseline CNN
6. Save the best checkpoint as `best_model.keras`
7. Build the Docker images
8. Start the services with Docker Compose
9. Validate predictions through the Streamlit interface

A fixed random seed is used throughout for reproducibility:

```python
SEED = 42
```

---

## 👩‍💻 Author

**Manahil Ishfaq**
*AI Engineer — Machine Learning & Deep Learning*

Focused on:
- Artificial Intelligence & Machine Learning
- Deep Learning & Computer Vision
- Explainable AI
- End-to-end AI Application Development

---

## 📄 License

This project is licensed under the **MIT License**.

```text
MIT License

Copyright (c) 2026 Manahil Ishfaq

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

<div align="center">

## ⭐ Project Highlights

**End-to-end Computer Vision + Explainable AI + Production Deployment**

| Capability | Implementation |
|---|---|
| 🌿 Disease Classification | Custom CNN (98.85% test accuracy) |
| 🖼️ Image Validation | CLIP zero-shot gate (`openai/clip-vit-base-patch32`) |
| 🔥 Explainability | Grad-CAM |
| 📜 Auditability | Persistent, searchable prediction history |
| ⚡ Inference API | FastAPI |
| 🎨 User Interface | Streamlit (dark, card-based UI) |
| 🐳 Containerization | Docker |
| 🔗 Service Orchestration | Docker Compose |
| 📊 Dataset | PlantVillage (54,305 images) |
| 🧠 Classes | 38 |
| 📐 Input Resolution | 224 × 224 |
| 🔬 Deep Learning Framework | TensorFlow / Keras |
| 🤖 Supporting Framework | PyTorch / Hugging Face Transformers |

*Built as a complete AI engineering project — from dataset preparation and comparative model evaluation, through explainability and input validation, to API development, containerization, and deployment.*

</div>
