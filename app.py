import sys
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Union

app = FastAPI(
    title="HepSense CDSS Core Engine - Tabular V2 (Root)",
    description="API for HepSense Longitudinal Tabular Liver Risk Assessment"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store ML artifacts
MODEL_PKG = None
SHAP_EXPLAINER = None

@app.on_event("startup")
def load_ml_assets():
    global MODEL_PKG, SHAP_EXPLAINER
    try:
        # Load the model package (check root and backend paths)
        pkg_path = "hepsense_temporal_xgb.joblib"
        if not os.path.exists(pkg_path):
            pkg_path = "../hepsense_temporal_xgb.joblib"
        MODEL_PKG = joblib.load(pkg_path)
        
        # Load the SHAP explainer
        shap_path = "shap_explainer.joblib"
        if not os.path.exists(shap_path):
            shap_path = "../shap_explainer.joblib"
        SHAP_EXPLAINER = joblib.load(shap_path)
        
        print("[SUCCESS] HepSense V2 Tabular ML models and SHAP engines loaded successfully.")
    except Exception as e:
        print(f"[CRITICAL] Failed to load ML assets: {str(e)}")


class LabHistoryInput(BaseModel):
    age: int = 55
    gender: str = "M"
    has_encephalopathy_mention: int = 0
    has_ascites_mention: int = 0
    has_variceal_bleeding_mention: int = 0
    # Dictionary mapping lab names to sequence of measurements (e.g. daily values)
    labs: Dict[str, List[float]]


# Clinician-friendly mapping for SHAP features
FEATURE_LABELS = {
    'age': 'Patient Age',
    'gender': 'Gender (Male)',
    'has_encephalopathy_mention': 'Hepatic Encephalopathy Note Mention',
    'has_ascites_mention': 'Ascites Note Mention',
    'has_variceal_bleeding_mention': 'Variceal Bleeding Note Mention',
    
    'Alanine_Aminotransferase_ALT_min': 'Min ALT Level',
    'Alanine_Aminotransferase_ALT_max': 'Max ALT Level',
    'Alanine_Aminotransferase_ALT_latest': 'Latest ALT Level',
    'Alanine_Aminotransferase_ALT_velocity': 'ALT Velocity (Rate of Change)',
    
    'Asparate_Aminotransferase_AST_min': 'Min AST Level',
    'Asparate_Aminotransferase_AST_max': 'Max AST Level',
    'Asparate_Aminotransferase_AST_latest': 'Latest AST Level',
    'Asparate_Aminotransferase_AST_velocity': 'AST Velocity (Rate of Change)',
    
    'Bilirubin_Total_min': 'Min Bilirubin Level',
    'Bilirubin_Total_max': 'Max Bilirubin Level',
    'Bilirubin_Total_latest': 'Latest Bilirubin Level',
    'Bilirubin_Total_velocity': 'Bilirubin Velocity (Rate of Change)',
    
    'Creatinine_min': 'Min Creatinine Level',
    'Creatinine_max': 'Max Creatinine Level',
    'Creatinine_latest': 'Latest Creatinine Level',
    'Creatinine_velocity': 'Creatinine Velocity (Rate of Change)',
    
    'INRPT_min': 'Min INR Level',
    'INRPT_max': 'Max INR Level',
    'INRPT_latest': 'Latest INR Level',
    'INRPT_velocity': 'INR Velocity (Rate of Change)',
    
    'Platelet_Count_min': 'Min Platelet Count',
    'Platelet_Count_max': 'Max Platelet Count',
    'Platelet_Count_latest': 'Latest Platelet Count',
    'Platelet_Count_velocity': 'Platelet Count Velocity (Rate of Change)'
}


@app.post("/predict_trajectory")
async def predict_trajectory(payload: LabHistoryInput):
    global MODEL_PKG, SHAP_EXPLAINER
    if MODEL_PKG is None or SHAP_EXPLAINER is None:
        raise HTTPException(status_code=503, detail="Machine learning models are not initialized.")
        
    try:
        # 1. Parse base inputs
        age = payload.age
        gender_num = 1 if payload.gender.upper() == "M" else 0
        has_encephalopathy = payload.has_encephalopathy_mention
        has_ascites = payload.has_ascites_mention
        has_variceal = payload.has_variceal_bleeding_mention
        
        # 2. Extract features from lab sequences
        extracted_features = {}
        
        lab_key_mapping = {
            "ALT": "Alanine_Aminotransferase_ALT",
            "Alanine_Aminotransferase_ALT": "Alanine_Aminotransferase_ALT",
            "AST": "Asparate_Aminotransferase_AST",
            "Asparate_Aminotransferase_AST": "Asparate_Aminotransferase_AST",
            "Bilirubin": "Bilirubin_Total",
            "Bilirubin_Total": "Bilirubin_Total",
            "Creatinine": "Creatinine",
            "INR": "INRPT",
            "INRPT": "INRPT",
            "Platelets": "Platelet_Count",
            "Platelet_Count": "Platelet_Count"
        }
        
        input_labs = {}
        for k, v in payload.labs.items():
            mapped_key = lab_key_mapping.get(k)
            if mapped_key:
                input_labs[mapped_key] = [float(x) for x in v if x is not None]

        # Extract features for all 6 standard tests
        standard_tests = [
            "Alanine_Aminotransferase_ALT",
            "Asparate_Aminotransferase_AST",
            "Bilirubin_Total",
            "Creatinine",
            "INRPT",
            "Platelet_Count"
        ]
        
        for test in standard_tests:
            vals = input_labs.get(test, [])
            
            if len(vals) == 0:
                extracted_features[f"{test}_min"] = np.nan
                extracted_features[f"{test}_max"] = np.nan
                extracted_features[f"{test}_latest"] = np.nan
                extracted_features[f"{test}_velocity"] = np.nan
            elif len(vals) == 1:
                extracted_features[f"{test}_min"] = vals[0]
                extracted_features[f"{test}_max"] = vals[0]
                extracted_features[f"{test}_latest"] = vals[0]
                extracted_features[f"{test}_velocity"] = 0.0
            else:
                extracted_features[f"{test}_min"] = min(vals)
                extracted_features[f"{test}_max"] = max(vals)
                extracted_features[f"{test}_latest"] = vals[-1]
                extracted_features[f"{test}_velocity"] = (vals[-1] - vals[0]) / (len(vals) - 1)

        # Assemble full input dictionary
        full_input = {
            "age": age,
            "gender": gender_num,
            "has_encephalopathy_mention": has_encephalopathy,
            "has_ascites_mention": has_ascites,
            "has_variceal_bleeding_mention": has_variceal,
            **extracted_features
        }
        
        # Convert to DataFrame
        df_features = pd.DataFrame([full_input])
        
        # Align columns to features expected by the model
        expected_features = MODEL_PKG['features']
        df_features = df_features.reindex(columns=expected_features)
        
        # Fill missing values (NaNs) with training medians
        medians = MODEL_PKG['medians']
        df_features = df_features.fillna(medians)

        # 3. Predict Probability
        calibrated_model = MODEL_PKG['calibrated_model']
        prob = float(calibrated_model.predict_proba(df_features)[0][1])
        optimal_threshold = MODEL_PKG.get('optimal_threshold', 0.05)

        # Define Risk Categories and Recommendations dynamically based on Youden optimal threshold
        risk_pct = prob * 100
        threshold_pct = optimal_threshold * 100
        
        if prob < (optimal_threshold / 2):
            risk_category = "Low Risk"
            recommendation = f"Patient is stable. Calibrated risk ({risk_pct:.1f}%) is below Youden J cutoff ({threshold_pct:.1f}%). Continue routine outpatient monitoring."
            color = "green"
            actions = ["Routine laboratory panel in 3 months", "Annual clinical ultrasound", "Supportive dietary counseling"]
        elif prob < optimal_threshold:
            risk_category = "Moderate Risk"
            recommendation = f"Moderate risk of decompensation. Calibrated risk ({risk_pct:.1f}%) is approaching Youden J cutoff ({threshold_pct:.1f}%). Repeat blood panel in 2 weeks."
            color = "orange"
            actions = ["Repeat liver panel in 14 days", "Evaluate for beta-blocker prophylaxis", "Dietary and lifestyle consultation"]
        else:
            risk_category = "Critical Risk"
            recommendation = f"CRITICAL RISK: Calibrated risk ({risk_pct:.1f}%) exceeds Youden J optimal clinical cutoff ({threshold_pct:.1f}%). Fast-track transplant evaluation."
            color = "red"
            actions = ["STAT inpatient admission / ICU standby", "Emergency Endoscopy (EGD) for varices screening", "Immediate Hepatology & Transplant consult"]

        # 4. Extract SHAP Explanations
        shap_explanation = SHAP_EXPLAINER(df_features)
        shap_vals = shap_explanation.values[0]
        
        shap_contributions = []
        for feat_name, val in zip(expected_features, shap_vals):
            label = FEATURE_LABELS.get(feat_name, feat_name)
            raw_val = float(df_features[feat_name].iloc[0])
            shap_contributions.append({
                "feature": feat_name,
                "label": label,
                "value": raw_val,
                "shap_value": float(val)
            })

        # Sort contributions by absolute SHAP values to identify top drivers
        sorted_contribs = sorted(shap_contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        
        # Format the top 3 drivers with human-readable text
        top_drivers = []
        for c in sorted_contribs[:3]:
            direction = "increases" if c["shap_value"] > 0 else "reduces"
            formatted_val = f"{c['value']:.2f}" if isinstance(c['value'], float) else str(c['value'])
            
            if "velocity" in c["feature"]:
                trend_desc = "rising" if c["value"] > 0 else "falling"
                top_drivers.append(
                    f"{c['label']} is {trend_desc} (value: {formatted_val}), which {direction} decompensation risk."
                )
            elif "mention" in c["feature"]:
                mention_desc = "present" if c["value"] == 1 else "absent"
                top_drivers.append(
                    f"{c['label']} is {mention_desc}, which {direction} decompensation risk."
                )
            else:
                top_drivers.append(
                    f"{c['label']} is {formatted_val}, which {direction} decompensation risk."
                )

        return {
            "risk_probability": risk_pct,
            "risk_category": risk_category,
            "recommendation": {
                "severity_level": risk_category.upper(),
                "recommendation": recommendation,
                "actions": actions,
                "color_code": color
            },
            "top_drivers": top_drivers,
            "shap_values": shap_contributions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
