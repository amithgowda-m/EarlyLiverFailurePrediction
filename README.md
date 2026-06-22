# HEPSENSE V3: Clinical Decision Support System (CDSS)

**An Explainable AI-driven Risk Stratification System for Liver Disease & Longitudinal Surveillance**

---

## 🧬 Project Overview
HepSense V3 is a medical-grade **Clinical Decision Support System (CDSS)** designed to predict the likelihood of liver disease and monitor patient trajectories longitudinally. It bridges the gap between raw biochemical laboratory test results and clinical action by combining:
1. An **Optuna-optimized XGBoost classifier** trained on the Indian Liver Patient Dataset (ILPD) and calibrated with **Platt Scaling** to output true clinical probabilities.
2. **SHAP Pathological Weighting** to provide transparent, explainable feature attributions at the point of care.
3. A **Serial Clinical Surveillance Timeline** that allows doctors to track risk trajectory day-by-day (e.g., over a 14-day stay) and restore past daily configurations with a single click.

---

## ⚠️ The Clinical Problem & Motivation
Traditional clinical models for liver function assessment (like standard LFT scores) suffer from:
* **Snapshot Bias:** Analyzing a single day's blood draw ignores the trajectory and rate of change of the patient's condition.
* **Lack of Transparency:** Many clinical machine learning models operate as "black boxes," providing scores without explaining which biological markers drove the prediction.
* **Static Scoring Limitations:** Traditional metrics lack dynamic patient surveillance. HepSense V3 tracks changes day-by-day, visualizes risk progression, and isolates critical abnormalities.

---

## 🏗️ System Architecture

### 1. Machine Learning Pipeline (`train_ilpd.py`)
* **Cohort Profile**: Trained on the UCI Indian Liver Patient Dataset (583 patients, 10 clinical features).
* **Optimization Protocol**: XGBoost Classifier tuned via a 60-trial **Optuna Bayesian optimization** study with 5-fold Stratified Cross-Validation.
* **Calibration Mapping**: Calibrated using **Platt Scaling** (Sigmoid calibration) so output risk probabilities match actual empirical disease likelihood.
* **Decision Boundary**: Optimized using **Youden’s J statistic** (optimal decision threshold: **0.7017**).
* **Performance**:
  - **Test ROC AUC**: **0.8044** (extremely strong discriminative power).
  - **Test PR AUC**: **0.9147**

### 2. FastAPI Backend (`backend/api.py`)
* **`GET /health`**: Enriched endpoint returning API status, active features, and complete clinical validation metadata dynamically.
* **`POST /predict`**: Accepts patient vitals and LFT values, checks normal ranges, computes the risk score, generates dynamic recommendations, and runs the SHAP explainer for attributions.

### 3. Academic Whiteboard UI (`frontend/src/App.jsx`)
* **Patient Chart (Left Sidebar)**: Input form supporting demographics, clinical reference case presets (Standard Baseline, Hepatitis Pattern, Advanced Cirrhosis), and full LFT inputs.
* **Diagnostic Whiteboard (Right Panel)**: Displays risk scores, abnormal laboratory flags, and recommended clinical actions.
* **Pathological Weighting**: Visual balance scale showing how each biomarker tilts the diagnosis toward healthy or pathological.
* **Serial Surveillance Timeline**: Stores sequential assessments (e.g. Day 1 &rarr; Day 14), enabling physicians to map out patient progression and click any node to restore historical lab values.

---

## 📂 Directory Structure
```
EarlyLiverFailurePrediction/
├── backend/
│   └── api.py                  # FastAPI Server
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # React Dashboard Layout & Timeline Track
│   │   └── index.css           # Academic Lora + Inter Theme Styling
│   ├── package.json
│   └── vite.config.js          # API Proxy configuration
├── ILPD/
│   └── indian_liver_patient.csv # Source Patient Cohort
├── ilpd_model_pkg.joblib       # Joblib Package (Model, Medians, Explainer)
├── train_ilpd.py               # XGBoost + Optuna Training Pipeline
├── pyproject.toml
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
* Python >= 3.12
* Node.js (v18+) & `npm`
* `uv` package manager

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amithgowda-m/EarlyLiverFailurePrediction.git
   cd EarlyLiverFailurePrediction
   ```

2. **Backend Setup:**
   Install PyTorch, XGBoost, SHAP, FastAPI, and other dependencies:
   ```bash
   uv sync
   ```

3. **Frontend Setup:**
   Install npm packages:
   ```bash
   cd frontend
   npm install
   ```

---

## 💻 Running the System

1. **Start the API Server:**
   From the project root:
   ```bash
   .venv\Scripts\python.exe -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Launch the Clinical Dashboard:**
   In a new terminal:
   ```bash
   cd frontend
   $env:PATH = "C:\nvm4w\nodejs;$env:PATH"; npm run dev
   ```

Access the dashboard in your browser at `http://localhost:5173`.
