"""
HepSense V3 — FastAPI Backend
Uses the ILPD (Indian Liver Patient Dataset) XGBoost classifier.
All paths are resolved from this file's location — no hardcoding.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

app = FastAPI(
    title="HepSense V3 — Liver Disease CDSS",
    description=(
        "Clinically-validated liver disease classifier trained on the "
        "Indian Liver Patient Dataset (ILPD). "
        "ROC AUC ~0.82 | Calibrated XGBoost | SHAP explanations."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ────────────────────────────────────────────────────────
MODEL_PKG: Optional[dict] = None


@app.on_event("startup")
def load_model() -> None:
    global MODEL_PKG
    path = os.path.join(ROOT, "ilpd_model_pkg.joblib")
    if not os.path.exists(path):
        print(f"[WARN] Model not found at {path}. Run train_ilpd.py first.")
        return
    MODEL_PKG = joblib.load(path)
    print(
        f"[OK] ILPD model loaded — "
        f"ROC AUC={MODEL_PKG['roc_auc']:.4f} | "
        f"Threshold={MODEL_PKG['optimal_threshold']:.4f} | "
        f"Features={len(MODEL_PKG['features'])}"
    )


# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURE_META = {
    "Age":                        {"label": "Patient Age",                   "unit": "years"},
    "Gender":                     {"label": "Biological Sex",                "unit": "M=1 / F=0"},
    "Total_Bilirubin":            {"label": "Total Bilirubin",               "unit": "mg/dL"},
    "Direct_Bilirubin":           {"label": "Direct (Conjugated) Bilirubin", "unit": "mg/dL"},
    "Alkaline_Phosphotase":       {"label": "Alkaline Phosphatase (ALP)",    "unit": "IU/L"},
    "Alamine_Aminotransferase":   {"label": "ALT (Alanine Aminotransferase)","unit": "U/L"},
    "Aspartate_Aminotransferase": {"label": "AST (Aspartate Aminotransferase)","unit":"U/L"},
    "Total_Protiens":             {"label": "Total Proteins",                "unit": "g/dL"},
    "Albumin":                    {"label": "Serum Albumin",                 "unit": "g/dL"},
    "Albumin_and_Globulin_Ratio": {"label": "Albumin / Globulin Ratio",      "unit": "ratio"},
}

NORMAL_RANGES = {
    "Total_Bilirubin":            (0.2, 1.2),
    "Direct_Bilirubin":           (0.0, 0.3),
    "Alkaline_Phosphotase":       (44, 147),
    "Alamine_Aminotransferase":   (7,  56),
    "Aspartate_Aminotransferase": (10, 40),
    "Total_Protiens":             (6.3, 8.2),
    "Albumin":                    (3.5, 5.0),
    "Albumin_and_Globulin_Ratio": (1.0, 2.5),
}


# ── Request / response schemas ────────────────────────────────────────────────
class PatientInput(BaseModel):
    age:                          int   = Field(45,   ge=1,   le=120, description="Patient age in years")
    gender:                       str   = Field("Male", description="'Male' or 'Female'")
    total_bilirubin:              float = Field(0.7,  ge=0.0, description="Total Bilirubin mg/dL")
    direct_bilirubin:             float = Field(0.1,  ge=0.0, description="Direct Bilirubin mg/dL")
    alkaline_phosphotase:         float = Field(187,  ge=0.0, description="ALP IU/L")
    alamine_aminotransferase:     float = Field(16,   ge=0.0, description="ALT U/L")
    aspartate_aminotransferase:   float = Field(18,   ge=0.0, description="AST U/L")
    total_protiens:               float = Field(6.8,  ge=0.0, description="Total Proteins g/dL")
    albumin:                      float = Field(3.3,  ge=0.0, description="Albumin g/dL")
    albumin_and_globulin_ratio:   float = Field(0.9,  ge=0.0, description="A/G Ratio")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    loaded = MODEL_PKG is not None
    return {
        "status":           "ok" if loaded else "degraded",
        "model_loaded":     loaded,
        "roc_auc":          round(MODEL_PKG["roc_auc"], 4) if loaded else None,
        "pr_auc":           round(MODEL_PKG["pr_auc"], 4) if loaded else None,
        "threshold":        round(MODEL_PKG["optimal_threshold"], 4) if loaded else None,
        "n_features":       len(MODEL_PKG["features"]) if loaded else 0,
        "dataset":          "ILPD — Indian Liver Patient Dataset (UCI/Kaggle)",
        "metadata": {
            "algorithm": "XGBoost Classifier (Optuna Tuned)",
            "dataset_name": "Indian Liver Patient Dataset (ILPD)",
            "n_samples": 583,
            "n_features": len(MODEL_PKG["features"]) if loaded else 10,
            "calibration": "Platt Scaling (Sigmoid)",
            "validation": "5-Fold Stratified Cross-Validation",
            "optimization": "Optuna Bayesian Optimization (60 trials)",
            "threshold_criterion": "Youden's J Statistic"
        } if loaded else None
    }


@app.post("/predict")
async def predict(patient: PatientInput):
    if MODEL_PKG is None:
        raise HTTPException(503, "Model not loaded. Run train_ilpd.py first.")

    try:
        gender_enc = 1 if patient.gender.lower().startswith("m") else 0

        raw = {
            "Age":                        float(patient.age),
            "Gender":                     float(gender_enc),
            "Total_Bilirubin":            patient.total_bilirubin,
            "Direct_Bilirubin":           patient.direct_bilirubin,
            "Alkaline_Phosphotase":       patient.alkaline_phosphotase,
            "Alamine_Aminotransferase":   patient.alamine_aminotransferase,
            "Aspartate_Aminotransferase": patient.aspartate_aminotransferase,
            "Total_Protiens":             patient.total_protiens,
            "Albumin":                    patient.albumin,
            "Albumin_and_Globulin_Ratio": patient.albumin_and_globulin_ratio,
        }

        df = pd.DataFrame([raw]).reindex(columns=MODEL_PKG["features"])
        df = df.fillna(MODEL_PKG["medians"])

        prob      = float(MODEL_PKG["model"].predict_proba(df)[0][1])
        threshold = MODEL_PKG["optimal_threshold"]
        is_liver  = prob >= threshold
        risk_pct  = round(prob * 100, 1)

        # Risk tier
        if prob < 0.35:
            tier        = "Low Risk"
            color       = "green"
            summary     = f"Biochemical profile suggests low likelihood of liver disease ({risk_pct}%). Routine annual follow-up recommended."
            actions     = [
                "Routine LFT repeat in 12 months",
                "Maintain healthy diet and avoid hepatotoxic substances",
                "Return if symptoms develop (jaundice, fatigue, abdominal pain)",
            ]
        elif prob < threshold:
            tier        = "Moderate Risk"
            color       = "orange"
            summary     = f"Borderline biochemical markers detected ({risk_pct}%). Closer monitoring warranted."
            actions     = [
                "Repeat liver function tests in 6–8 weeks",
                "Hepatology referral for further evaluation",
                "Liver ultrasound recommended",
            ]
        else:
            tier        = "High Risk — Liver Disease Likely"
            color       = "red"
            summary     = f"Biochemical profile is consistent with liver disease ({risk_pct}%). Immediate hepatology review required."
            actions     = [
                "Urgent Hepatology consultation",
                "Abdominal ultrasound + fibroscan",
                "Review medications for hepatotoxic drugs",
                "Consider liver biopsy if imaging inconclusive",
            ]

        # Which markers are outside normal range
        abnormal_flags = []
        for feat, (lo, hi) in NORMAL_RANGES.items():
            val = raw.get(feat, None)
            if val is not None:
                label = FEATURE_META[feat]["label"]
                unit  = FEATURE_META[feat]["unit"]
                if val < lo:
                    abnormal_flags.append({"feature": feat, "label": label, "value": val, "unit": unit, "direction": "low", "normal": f"{lo}–{hi}"})
                elif val > hi:
                    abnormal_flags.append({"feature": feat, "label": label, "value": val, "unit": unit, "direction": "high", "normal": f"{lo}–{hi}"})

        # SHAP
        shap_contributions = []
        shap_exp = MODEL_PKG.get("shap_explainer")
        if shap_exp is not None:
            try:
                sv = shap_exp.shap_values(df)
                # shap_values returns list[array] for binary; take class-1 values
                sv_arr = sv[1] if isinstance(sv, list) else sv
                for feat, val, sv_val in zip(MODEL_PKG["features"], df.iloc[0], sv_arr[0]):
                    shap_contributions.append({
                        "feature":    feat,
                        "label":      FEATURE_META.get(feat, {}).get("label", feat),
                        "value":      round(float(val), 3),
                        "shap_value": round(float(sv_val), 4),
                        "unit":       FEATURE_META.get(feat, {}).get("unit", ""),
                    })
                shap_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            except Exception as shap_err:
                print(f"[SHAP warn] {shap_err}")

        return {
            "probability":        risk_pct,
            "prediction":         "Liver Disease" if is_liver else "Healthy",
            "tier":               tier,
            "color":              color,
            "summary":            summary,
            "actions":            actions,
            "abnormal_markers":   abnormal_flags,
            "shap_contributions": shap_contributions,
            "model_metrics": {
                "roc_auc":   round(MODEL_PKG["roc_auc"], 4),
                "pr_auc":    round(MODEL_PKG["pr_auc"], 4),
                "threshold": round(threshold, 4),
            },
        }

    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
