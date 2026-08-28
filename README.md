# VisionAI

## AI-Powered Image Quality & Defect Detection

VisionAI is a full-stack AI application for automated image-quality assessment and conservative visual defect detection.

The system accepts an uploaded image, extracts interpretable computer-vision features, applies a locally trained machine-learning model, and returns:

- Quality score
- Quality label
- Predicted quality class
- Confidence
- Severity
- Detected issues
- Image statistics
- Analysis history

VisionAI combines **Computer Vision + Machine Learning** rather than relying only on fixed image-processing thresholds.

All image processing and ML inference are performed locally.

**No external AI/vision API or API key is required.**

---

# Quick Start - Run & Test VisionAI

The recommended way to evaluate VisionAI is with **Docker Compose**.

Docker runs the complete application:

```text
React Frontend
      +
FastAPI Backend
      +
Random Forest Model
      +
SQLite Database
```

## Prerequisites

Install:

- Docker Desktop
- Docker Compose

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd VisionAI
```

> Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the final repository URL before submission.

## 2. Build the Application

From the project root:

```bash
docker compose build
```

## 3. Start the Application

```bash
docker compose up
```

Wait until the frontend and backend containers are running.

## 4. Open the Application

### Frontend

```text
http://localhost:5173
```

### Backend

```text
http://localhost:8001
```

### FastAPI Documentation

```text
http://localhost:8001/docs
```

### Health Check

```text
http://localhost:8001/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "VisionAI API",
  "version": "1.0.0"
}
```

## 5. Test the Application

Representative images are included in:

```text
sample_images/
```

Available examples:

```text
clean.jpg
blur.jpg
underexposed.jpg
overexposed.jpg
noise.jpg
degraded.jpg
```

To test through the frontend:

1. Open `http://localhost:5173`
2. Upload an image from `sample_images/`
3. Preview the image
4. Click **Analyze Image**
5. Review the quality assessment
6. View the saved result in **Analysis History**

## 6. Stop the Application

```bash
docker compose down
```

Analysis history is stored in a persistent Docker volume and survives normal container recreation.

> `docker compose down -v` removes the database volume and its stored history.

---

# System Architecture

```text
+----------------------+
|        USER          |
+----------+-----------+
           |
           | Upload Image
           v
+----------------------+
|    React Frontend    |
|     Port 5173        |
|                      |
| Upload / Preview     |
| Results / History    |
+----------+-----------+
           |
           | REST API
           v
+----------------------+
|    FastAPI Backend   |
|      Port 8001       |
+----------+-----------+
           |
           v
+----------------------+
| Input Validation &   |
| Image Decoding       |
+----------+-----------+
           |
     +-----+-----+
     |           |
     v           v
+---------+  +-------------+
|Computer |  |Random Forest|
| Vision  |  | ML Model    |
|Features |  | Prediction  |
+----+----+  +------+------+
     |              |
     +------+-------+
            |
            v
+----------------------+
|   Decision Engine    |
|                      |
| ML Prediction        |
| + CV Evidence        |
| + Local Anomaly      |
+----------+-----------+
           |
           v
+----------------------+
| Structured Result    |
|                      |
| Score / Label        |
| Confidence           |
| Severity / Issues    |
| Statistics           |
+----------+-----------+
           |
     +-----+-----+
     |           |
     v           v
+---------+  +-------------+
| React   |  | SQLite      |
| Results |  | History     |
+---------+  +-------------+
```

The **Random Forest model is the primary learned global quality classifier**.

Computer-vision measurements provide interpretable features and supporting evidence for the final decision.

---

# Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| Backend | Python 3.12, FastAPI, Uvicorn |
| Computer Vision | OpenCV, Pillow, NumPy |
| Machine Learning | scikit-learn, Random Forest, Joblib, pandas |
| Database | SQLite, SQLAlchemy |
| Deployment | Docker, Docker Compose |

---

# Features

VisionAI supports:

- Image upload and preview
- Drag-and-drop upload
- JPG, JPEG, PNG, and WEBP images
- Maximum upload size of 10 MB
- AI-based quality classification
- Blur detection
- Underexposure detection
- Overexposure detection
- Noise detection
- Severe degradation detection
- Conservative potential visual defect detection
- Quality score from 0 to 100
- Confidence score
- Severity level
- Detected issues
- Explainable image statistics
- Persistent analysis history
- Expandable history details
- Individual history deletion
- Clear History
- REST API
- Health endpoint
- Docker deployment

Possible final labels are:

```text
ACCEPTABLE
DEGRADED
POTENTIALLY_DEFECTIVE
```

---

# AI / ML Approach

VisionAI uses a hybrid approach:

```text
Image
  |
  v
Computer Vision Feature Extraction
  |
  v
18 Engineered Features
  |
  v
Random Forest Classifier
  |
  +----------------------+
  |                      |
  v                      v
Global Prediction    Supporting CV /
                     Local Anomaly
  |                      |
  +----------+-----------+
             |
             v
       Final Decision
```

The learned Random Forest model is responsible for the primary global image-quality prediction.

Computer-vision evidence is used to:

- Make the model input interpretable
- Support strong degradation decisions
- Provide image statistics
- Conservatively identify unusual localized regions

This ensures that the final assessment is meaningfully based on machine learning rather than only manually selected thresholds.

---

# Why Random Forest?

The model operates on engineered numerical image-quality features rather than raw pixels.

Random Forest was selected because it:

- Performs well on tabular numerical features
- Learns nonlinear relationships between features
- Works effectively with a relatively small dataset
- Provides feature importance
- Has fast inference
- Does not require GPU hardware
- Is lightweight to deploy
- Supports reproducible training

This provides a practical balance between **ML capability, explainability, and deployment simplicity**.

---

# Computer Vision Features

The final global classifier uses **18 engineered image-quality features**.

## Brightness and Exposure

- Brightness
- Dark pixel ratio
- Bright pixel ratio
- Extreme pixel ratio

## Sharpness and Gradients

- Sharpness
- Gradient mean
- Gradient standard deviation
- Strong edge ratio
- Laplacian extreme ratio

## Contrast

- Contrast
- Local contrast mean
- Local contrast standard deviation

## Noise and Texture

- Noise level
- Entropy
- Edge density

## Local Intensity

- Patch intensity standard deviation
- Patch intensity range

## Color

- Saturation

The API additionally reports image width, height, and channel count for explainability.

---

# Dataset

The training data combines:

- **250 high-quality images** selected from the public KonIQ-10k image-quality dataset
- **20 additional source images**
- Controlled synthetic degradations

Total source images:

```text
250 KonIQ images
+ 20 additional images
----------------------
270 source images
```

KonIQ selection metadata is stored in:

```text
backend/dataset/koniq_selection.csv
```

The additional source images are stored in:

```text
backend/dataset/source/
```

Large generated datasets and the local KonIQ working-image directory are intentionally excluded from the repository.

The scripts required to reproduce the data-generation process are included.

---

# Synthetic Degradation Generation

Controlled transformations were used to generate known image-quality conditions.

Generated categories included:

- Clean
- Blur
- Underexposed
- Overexposed
- Noise
- Degraded
- Synthetic visual defect

The final global classifier uses six classes:

```text
blur
clean
degraded
noise
overexposed
underexposed
```

Synthetic transformations provide known labels while allowing controlled evaluation of common image-quality problems.

Dataset generation is implemented in:

```text
backend/ml/generate_dataset.py
```

---

# Train/Test Separation

The dataset is split at the **source-image level before synthetic degradations are generated**.

This prevents different degraded versions of the same original image from appearing in both training and testing data.

The deterministic split uses:

```text
Random seed: 42

Total source images: 270
Training sources:    216
Testing sources:      54
```

For the six-class global model:

```text
Training samples: 1296
Testing samples:   324

Training samples per class: 216
Testing samples per class:   54
```

This provides a more meaningful evaluation on degradations generated from **unseen source images** and reduces data leakage.

---

# Model Training

The final global classifier is a Random Forest configured with:

```text
Estimators:          300
Random state:        42
Class weighting:     balanced
Parallel processing: enabled
```

Training script:

```text
backend/ml/train.py
```

Final trained model:

```text
backend/artifacts/quality_model_6class.joblib
```

Feature metadata:

```text
backend/artifacts/feature_metadata_6class.json
```

Evaluation report:

```text
backend/artifacts/evaluation_report_6class.txt
```

---

# Model Evaluation

The final model was evaluated on:

```text
324 held-out test samples
```

generated from the 54 test source images that were not used for training.

## Overall Metrics

| Metric | Result |
|---|---:|
| Accuracy | **88.89%** |
| Balanced Accuracy | **88.89%** |
| Macro F1 | **88.68%** |
| Weighted F1 | **88.68%** |

## Per-Class Results

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Blur | 94.55% | 96.30% | 95.41% |
| Clean | 84.09% | 68.52% | 75.51% |
| Degraded | 94.44% | 94.44% | 94.44% |
| Noise | 85.45% | 87.04% | 86.24% |
| Overexposed | 81.36% | 88.89% | 84.96% |
| Underexposed | 92.98% | 98.15% | 95.50% |

The strongest performance is observed for blur, severe degradation, and underexposure.

The clean class is more challenging because some moderately degraded images can have feature distributions similar to clean images.

---

# Feature Importance and Explainability

Random Forest feature importance provides insight into which image characteristics contribute most strongly to classification.

Some of the strongest learned features include:

```text
Sharpness
Laplacian Extreme Ratio
Bright Pixel Ratio
Noise Level
Gradient Standard Deviation
```

These correspond to meaningful image characteristics such as:

- Focus
- Exposure
- Noise
- Edge structure
- High-frequency detail

The frontend also displays:

- Quality score
- Quality label
- Predicted class
- Confidence
- Severity
- Detected issues
- Image statistics

This makes the result more interpretable than returning only a class label.

---

# Potential Visual Defect Detection

Localized defects were investigated separately from global image-quality degradation.

Two experimental approaches were evaluated:

1. Patch-based local anomaly scoring
2. Binary Random Forest defect classification using global and local features

The patch-based approach produced limited held-out defect recall.

The experimental binary classifier detected some defects but produced too many false positives on clean images.

Because these approaches did not generalize reliably enough, they were **not promoted to the primary AI classifier**.

The final system therefore uses localized anomaly information only as a **conservative supporting warning signal**.

A potential defect warning is considered only when:

- The global model strongly considers the image clean
- No strong global degradation signal exists
- Local anomaly evidence is unusually high

Therefore:

> Potential visual defect detection should be interpreted as a conservative localized anomaly warning, not as a fully validated industrial defect classifier.

Experimental reports are retained at:

```text
backend/artifacts/local_defect_evaluation.txt
backend/artifacts/defect_binary_evaluation.txt
```

---

# Final Decision Logic

The final decision follows four simple cases.

### 1. ML-Detected Degradation

If the Random Forest predicts:

```text
blur
underexposed
overexposed
noise
degraded
```

the image is labeled:

```text
DEGRADED
```

### 2. Strong CV Evidence

If the ML model predicts clean but very strong computer-vision degradation evidence exists, the result can be overridden to:

```text
DEGRADED
```

### 3. Conservative Localized Anomaly

If the image is strongly predicted as globally clean, has weak global degradation evidence, but contains an unusually strong localized anomaly, the system may return:

```text
POTENTIALLY_DEFECTIVE
```

### 4. Acceptable

Otherwise:

```text
ACCEPTABLE
```

The ML model remains the primary global decision mechanism.

---

# REST API

The backend is implemented using FastAPI.

Base address:

```text
http://localhost:8001
```

Interactive API documentation:

```text
http://localhost:8001/docs
```

## Health Check

```http
GET /health
```

## Analyze Image

```http
POST /api/analyze
```

Example:

```bash
curl -X POST -F "file=@sample_images/clean.jpg" http://localhost:8001/api/analyze
```

Example response structure:

```json
{
  "filename": "clean.jpg",
  "quality_score": 99,
  "quality_label": "ACCEPTABLE",
  "predicted_class": "clean",
  "confidence": 0.9567,
  "severity": "LOW",
  "issues": [],
  "probabilities": {},
  "image_statistics": {},
  "analysis_id": 1
}
```

## Analysis History

```http
GET /api/history
```

Optional limit:

```text
/api/history?limit=20
```

## Delete History Record

```http
DELETE /api/history/{analysis_id}
```

## Clear History

```http
DELETE /api/history
```

All endpoints can also be tested through:

```text
http://localhost:8001/docs
```

---

# Database and Persistence

VisionAI uses:

```text
SQLite + SQLAlchemy
```

Analysis records store information including:

- Filename
- Quality score
- Quality label
- Predicted class
- Confidence
- Severity
- Issues
- Probabilities
- Image statistics
- Creation time

For local execution, the database is automatically created at:

```text
backend/visionai.db
```

No external database server is required.

When using Docker Compose:

```text
DATABASE_PATH=/app/data/visionai.db
```

is used.

The database is stored in the Docker named volume:

```text
visionai_data
```

so analysis history survives normal container recreation.

---

# Model Loading and Inference After Deployment

The trained Random Forest model is stored at:

```text
backend/artifacts/quality_model_6class.joblib
```

The model artifact is packaged inside the backend Docker image.

When FastAPI starts, the model service automatically loads the serialized Joblib model and validates the expected feature configuration and model classes.

For every uploaded image:

```text
Upload
  |
  v
Validate File
  |
  v
Decode Image
  |
  v
Extract 18 CV Features
  |
  v
Random Forest Inference
  |
  v
Class Probabilities
  |
  v
Supporting CV / Local Analysis
  |
  v
Final Quality Decision
  |
  v
Structured JSON
  |
  +------> React Frontend
  |
  `------> SQLite History
```

No model retraining or model download is required after deployment.

No external AI service is contacted during inference.

---

# Sample Images

Representative examples are provided in:

```text
sample_images/
```

| File | Condition |
|---|---|
| `clean.jpg` | Clean / acceptable |
| `blur.jpg` | Blur |
| `underexposed.jpg` | Underexposure |
| `overexposed.jpg` | Overexposure |
| `noise.jpg` | Noise |
| `degraded.jpg` | Severe degradation |

These samples provide a quick way for an examiner to test different quality conditions through the deployed application.

---

# Input Validation and Error Handling

The backend accepts:

```text
JPG
JPEG
PNG
WEBP
```

Maximum upload size:

```text
10 MB
```

The API validates uploaded data before ML inference and handles:

- Unsupported formats
- Oversized uploads
- Invalid image data
- Unreadable images
- Missing history records
- Analysis errors

Appropriate HTTP status codes and structured error responses are returned.

---

# Project Structure

```text
VisionAI/
|
|-- backend/
|   |
|   |-- app/
|   |   |-- api/
|   |   |-- core/
|   |   |-- db/
|   |   |-- schemas/
|   |   `-- services/
|   |
|   |-- artifacts/
|   |   |-- quality_model_6class.joblib
|   |   |-- feature_metadata_6class.json
|   |   |-- evaluation_report_6class.txt
|   |   |-- local_defect_evaluation.txt
|   |   `-- defect_binary_evaluation.txt
|   |
|   |-- dataset/
|   |   |-- source/
|   |   `-- koniq_selection.csv
|   |
|   |-- ml/
|   |-- tests/
|   |-- uploads/
|   |-- requirements.txt
|   `-- .env.example
|
|-- frontend/
|   |-- public/
|   `-- src/
|       |-- assets/
|       |-- services/
|       |-- App.jsx
|       |-- App.css
|       `-- main.jsx
|
|-- docker/
|   |-- backend.Dockerfile
|   `-- frontend.Dockerfile
|
|-- sample_images/
|   |-- clean.jpg
|   |-- blur.jpg
|   |-- underexposed.jpg
|   |-- overexposed.jpg
|   |-- noise.jpg
|   `-- degraded.jpg
|
|-- docker-compose.yml
|-- README.md
|-- LICENSE
|-- .gitignore
`-- .dockerignore
```

---

# Reproducing the ML Pipeline

ML utilities are located in:

```text
backend/ml/
```

The main scripts are:

| Script | Purpose |
|---|---|
| `prepare_koniq_subset.py` | Prepare the selected public dataset subset |
| `generate_dataset.py` | Generate controlled degradation samples |
| `build_features.py` | Extract model features |
| `train.py` | Train and evaluate the final Random Forest |
| `evaluate_local_defect.py` | Evaluate localized anomaly detection |
| `train_defect_binary.py` | Experimental binary defect classifier |

The final trained model is already included in the repository, so reproducing training is **not required to run the application**.

---

# Local Development - Optional

Docker Compose is the recommended evaluation method.

For development without Docker:

## Backend

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r .\backend\requirements.txt
```

Start the backend:

```powershell
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:5173
```

---

# Known Limitations

VisionAI is a technical assessment project rather than a production industrial inspection system.

Current limitations include:

- The global model is trained primarily on controlled synthetic degradations.
- Real-world distortion distributions may differ from the training data.
- The clean class is more difficult to distinguish than several degradation classes.
- Moderately overexposed images can occasionally overlap with clean feature distributions.
- Localized defect detection is conservative and experimental.
- Small or subtle defects may not trigger a potential-defect warning.
- Random Forest confidence values have not undergone dedicated probability calibration.
- A larger real-world annotated defect dataset would be required for production-grade defect detection.

These limitations are documented to avoid overstating the model's capabilities.

---

# Reproducibility

The project supports reproducibility through:

- Fixed random seed (`42`)
- Source-level train/test separation
- Stored trained model
- Stored feature metadata
- Stored evaluation reports
- Stored training/testing feature tables
- Version-pinned Python dependencies
- Dataset-generation scripts
- Training scripts
- Dockerized backend and frontend
- Docker Compose configuration
- Documented limitations

Important artifacts include:

```text
backend/artifacts/quality_model_6class.joblib
backend/artifacts/feature_metadata_6class.json
backend/artifacts/evaluation_report_6class.txt
backend/artifacts/train_features.csv
backend/artifacts/test_features.csv
```

---

# Privacy and External Services

VisionAI performs computer-vision processing and machine-learning inference locally.

The application does **not** require:

- OpenAI API
- Google Vision API
- AWS Rekognition
- Azure Computer Vision
- External LLM services
- Cloud-hosted AI inference

No external AI API key is required.

---

# Author

**Nalivela Abhinaya**

VisionAI - AI-Powered Image Quality & Defect Detection