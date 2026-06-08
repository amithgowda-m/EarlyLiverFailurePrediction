# HepSense: Modular Clinical Decision Support System (CDSS)

**An AI-driven Risk Stratification System for Early Liver Failure Prediction**

<div align="center">
  <em>Shifting the clinical paradigm from static "disease staging" to dynamic, longitudinal future prediction.</em>
</div>

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [The Clinical Problem](#-the-clinical-problem)
- [System Architecture](#-system-architecture)
  - [1. Vision Expert (DANN + DenseNet121)](#1-vision-expert-dann--densenet121)
  - [2. Clinical Temporal Expert (T-MELD)](#2-clinical-temporal-expert-t-meld)
  - [3. Decision-Level Fusion Engine](#3-decision-level-fusion-engine)
- [Results & Performance](#-results--performance)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
- [Running the System](#-running-the-system)
- [Expected Outcomes & Impact](#-expected-outcomes--impact)

---

## 🧬 Project Overview

Liver cirrhosis is the 11th leading cause of death globally (over 1.3 million deaths annually). 
**HepSense** is designed as a multi-modal framework that integrates both computer vision (for medical imaging analysis) and time-series extraction (for longitudinal electronic health records) to provide a holistic, dynamic risk assessment of liver cirrhosis. It acts as a temporal AI "early warning radar" to reduce preventable ICU mortality by predicting acute decompensation events before they occur.

## ⚠️ The Clinical Problem

Current clinicians rely on outdated, static scoring equations (e.g., MELD-Na, FIB-4). 
* **Snapshot Bias:** Static scores analyze a single day's blood test, creating a dangerous bias that misses declining patient trajectories.
* **Blind Spots:** Clinicians are often blind to impending, fatal crises like variceal hemorrhage or encephalopathy.
* **Reactive vs. Proactive:** Most existing models predict mortality *after* a patient enters critical condition in the ICU. HepSense bridges this gap by forecasting the actual future onset of an emergency.

---

## 🏗 System Architecture

HepSense uses a **"Chief Medical Officer" approach**, deploying independent AI experts and unifying them via decision-level fusion.

### 1. Vision Expert (DANN + DenseNet121)
A deep learning computer vision model built to classify the progression of liver cirrhosis stages (F0 to F4) from 2D B-mode ultrasound imaging.
* **Architecture:** `DenseNet121` feature extractor integrated with a **Domain-Adversarial Neural Network (DANN)**. 
* **Texture over Artifacts:** The DANN framework utilizes a Gradient Reversal Layer (GRL) to force the network to map structural parenchyma texture rather than variations in machine calibration or acoustic gain.
* **Explainable AI:** Grad-CAM heatmaps highlight cirrhotic nodules to verify that the network focuses on anatomical pathology.

### 2. Clinical Temporal Expert (T-MELD)
A quantitative data-mining structure utilizing multi-modal clinical histories to capture the velocity and acceleration of a patient's decline.
* **Temporal Engineering:** Extracts 34 longitudinal features using `tsfresh` to calculate biomarker velocity (e.g., the rate of bilirubin increase).
* **Predictive Modeling:** Time-Series Gradient Boosted Trees (`XGBoost`) optimized via `Optuna` and calibrated with Isotonic Regression.
* **Explainability (XAI):** Employs SHAP (SHapley Additive exPlanations) values to visualize feature importance, ensuring absolute medical transparency and physician trust.

### 3. Decision-Level Fusion Engine
The integration layer written in `FastAPI`. It merges the probabilistic outputs of the Vision and Temporal experts. It incorporates rule-based safety nets to catch clinical discordance (e.g., flagging a high-risk alert if a patient has an F0 liver image but rapidly accelerating lab failure trends).

---

## 📊 Results & Performance

* **Vision Pipeline:** Achieved **98.66% validation accuracy** across the 5-stage fibrosis dataset, with a 0.97 F1-score discriminating F2 vs. F3 stages.
* **Clinical Pipeline (T-MELD):** Achieved an **AUROC of 0.6891**, statistically outperforming the standard MELD baseline of 0.6834 on highly imbalanced clinical cohorts. Negative Predictive Value securely clears stable patients with 98% precision.

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
   Sync the Python environment to install PyTorch, XGBoost, FastAPI, and other dependencies.
   ```bash
   uv sync
   ```

3. **Frontend Setup:**
   Install the React.js dependencies.
   ```bash
   cd frontend
   npm install
   ```

---

## 💻 Running the System

The HepSense platform requires running the FastAPI backend and the React frontend concurrently.

1. **Start the API Server:**
   From the project root:
   ```bash
   cd backend
   python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Launch the Clinical Dashboard:**
   In a new terminal:
   ```bash
   cd frontend
   npm run dev
   ```

Access the dashboard at `http://localhost:5173`. 

*(Note: Production model weights and raw datasets are securely tracked outside of this repository due to size constraints. Ensure you have the `.pth` and `.joblib` artifacts in the root folder before inferencing).*

---

## 🎯 Expected Outcomes & Impact

* **Performance Shift:** Establishing proven, quantitative superiority over the static MELD clinical score.
* **Clinical Impact:** Empowers hepatologists to schedule preemptive, prophylactic interventions safely (e.g., executing an endoscopy before a fatal hemorrhage occurs), optimizing hospital triage resources.
* **Trust & Transparency:** Decision-level fusion combined with SHAP and Grad-CAM directly tackles the "black box" liability of medical AI, prioritizing explainability for clinical deployment.
