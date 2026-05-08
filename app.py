import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.utilities.dataframe_functions import impute
import io
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CLINICAL TRANSLATION DICTIONARY
# -----------------------------------------------------------------------------
CLINICAL_MAP = {
    'time_reversal_asymmetry_statistic': 'Acute Trajectory Volatility',
    'linear_trend': 'Linear Degradation Trend',
    'energy_ratio_by_chunks': 'Intermittent Spikes',
    'fft_coefficient': 'Cyclical Fluctuation',
    'quantile': 'Distribution Shift',
    'cwt_coefficients': 'Wavelet Shift',
    'ar_coefficient': 'Autoregressive Drift',
    'agg_linear_trend': 'Aggregated Trend',
    'spkt_welch_density': 'Spectral Density',
    'variance': 'Variance',
    'standard_deviation': 'Standard Deviation',
    'maximum': 'Maximum Level',
    'minimum': 'Minimum Level',
    'mean': 'Average Level',
    'median': 'Median Level',
    'sum_values': 'Cumulative Burden',
    'abs_energy': 'Absolute Energy',
    'mean_abs_change': 'Mean Absolute Change',
    'mean_change': 'Mean Change',
    'binned_entropy': 'Binned Entropy',
    'approximate_entropy': 'Approximate Entropy',
    'sample_entropy': 'Sample Entropy',
    'count_above_mean': 'Spikes Above Baseline',
    'count_below_mean': 'Drops Below Baseline',
    'last_location_of_minimum': 'Timing of Lowest Drop',
    'last_location_of_maximum': 'Timing of Highest Peak',
    'longest_strike_below_mean': 'Longest Period Below Baseline',
    'longest_strike_above_mean': 'Longest Period Above Baseline',
    'number_crossing_m': 'Baseline Crossings',
    'percentage_of_reoccurring_values_to_all_values': 'Reoccurring Value Ratio',
    'ratio_value_number_to_time_series_length': 'Data Point Density',
    'index_mass_quantile': 'Early vs Late Burden Shift'
}

def translate_feature_name(raw_name):
    clean_name = raw_name
    for key, val in CLINICAL_MAP.items():
        if key in clean_name:
            clean_name = clean_name.replace(key, val)
    clean_name = clean_name.replace('__', ' ').replace('_', ' ').replace('attr "real" ', '').replace('attr "imag" ', '').replace('attr "angle" ', '')
    # Remove extraneous coefficient numbers if present to keep it strictly clinical
    import re
    clean_name = re.sub(r'coeff \d+', '', clean_name)
    clean_name = re.sub(r'q \d+\.\d+', '', clean_name)
    return clean_name.strip()

# -----------------------------------------------------------------------------
# 1. Initialization & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="t-MELD: Temporal Liver Risk Stratification",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI polish and a frictionless clinical experience
st.markdown("""
<style>
    .metric-container {
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-title { font-size: 20px; font-weight: 600; color: #555; letter-spacing: 1px;}
    .metric-value { font-size: 72px; font-weight: 800; margin: 10px 0; }
    .risk-low { background-color: #f0fdf4; border-left: 8px solid #22c55e; color: #166534; }
    .risk-medium { background-color: #fefce8; border-left: 8px solid #eab308; color: #854d0e; }
    .risk-high { background-color: #fef2f2; border-left: 8px solid #ef4444; color: #991b1b; }
    .summary-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 8px;
        font-size: 18px;
        color: #334155;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

st.title("t-MELD: Temporal Liver Risk Stratification")
st.markdown("Clinical Decision Support System (CDSS) for predicting 14-day hepatic decompensation using calibrated sequence modeling.")

@st.cache_resource
def load_models():
    """Load calibrated models and rigorously pruned feature set."""
    try:
        model = joblib.load('tmeld_production.pkl')
        selected_cols = joblib.load('selected_features.pkl')
        
        try:
            le = joblib.load('le_production.pkl')
        except FileNotFoundError:
            le = None
            
        static_feats = ['age', 'gender']
        ts_cols = [c for c in selected_cols if c not in static_feats]
        
        kind_to_fc_parameters = from_columns(ts_cols)
        
        return model, selected_cols, le, kind_to_fc_parameters
    except FileNotFoundError:
        return None, None, None, None

model, selected_cols, le, kind_to_fc_parameters = load_models()

if model is None:
    st.error("⚠️ **System Offline:** Production artifacts (`tmeld_production.pkl` or `selected_features.pkl`) not found. Please run `train.py`.")
    st.stop()

# Extract underlying XGBoost model for SHAP from CalibratedClassifierCV
if hasattr(model, 'calibrated_classifiers_'):
    base_xgb_model = model.calibrated_classifiers_[0].estimator
else:
    base_xgb_model = model

# -----------------------------------------------------------------------------
# Patient Demographics (Sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Patient Demographics")
    st.markdown("Enter static clinical data.")
    age_val = st.number_input("Patient Age", min_value=18, max_value=120, value=55)
    
    if le is not None:
        gender_val = st.selectbox("Gender", le.classes_)
    else:
        gender_val = st.selectbox("Gender", ["M", "F"])

# -----------------------------------------------------------------------------
# 2. The "Zero-Friction" Data Input Zone
# -----------------------------------------------------------------------------
st.header("Longitudinal EHR Import")
st.markdown("Input the patient's irregular clinical timeline. Missing days will be automatically imputed via forward-filling.")

target_tests = [
    "Bilirubin, Total", "INR(PT)", "Creatinine", 
    "Platelet Count", "Alanine Aminotransferase (ALT)", 
    "Asparate Aminotransferase (AST)"
]

tab1, tab2 = st.tabs(["📁 EHR File Upload", "📋 Quick Paste"])

df_input = None

with tab1:
    uploaded_file = st.file_uploader(
        "Upload EHR CSV Extract", 
        type=["csv"],
        help="CSV must contain a `charttime` column and test name columns."
    )
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success("EHR File successfully parsed!")
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")

with tab2:
    pasted_data = st.text_area(
        "Paste EHR Data (Tab-Separated or CSV)",
        height=150,
        help="Paste directly from your EHR system grid.",
        placeholder='charttime,"Bilirubin, Total",INR(PT),Creatinine\n2023-10-01,1.2,1.1,0.9\n2023-10-05,2.1,1.4,1.2'
    )
    if pasted_data and uploaded_file is None:
        try:
            sep = ',' if ',' in pasted_data.split('\n')[0] else '\t'
            df_input = pd.read_csv(io.StringIO(pasted_data), sep=sep)
            st.success("Pasted data successfully parsed!")
        except Exception as e:
            st.error("Error reading pasted data. Ensure it includes headers and is formatted correctly.")

if df_input is not None and not df_input.empty:
    st.dataframe(df_input.head(3), use_container_width=True)

# -----------------------------------------------------------------------------
# 3 & 4. Automated Preprocessing & Optimized Inference Engine
# -----------------------------------------------------------------------------
if st.button("Run Calibrated Risk Engine", type="primary", use_container_width=True):
    if df_input is None or df_input.empty:
        st.warning("Please upload a file or paste EHR data before running the engine.")
        st.stop()
        
    with st.spinner("Executing Clinical Preprocessing and Time-Series Inference..."):
        try:
            cols_lower = {c.lower(): c for c in df_input.columns}
            if 'charttime' not in cols_lower:
                st.error("CRITICAL ERROR: Input data must contain a `charttime` column.")
                st.stop()
            else:
                df_input = df_input.rename(columns={cols_lower['charttime']: 'charttime'})
            
            df_input['charttime'] = pd.to_datetime(df_input['charttime'], errors='coerce')
            df_input = df_input.dropna(subset=['charttime'])
            
            available_tests = [c for c in target_tests if c in df_input.columns]
            if not available_tests:
                st.error(f"CRITICAL ERROR: None of the target clinical tests found. Expected at least one of: {target_tests}")
                st.stop()
                
            df_input = df_input.sort_values('charttime')
            
            # Resample to 14-day uniform grid (Daily) and Forward Fill
            df_input.set_index('charttime', inplace=True)
            df_daily = df_input[available_tests].resample('D').mean()
            df_daily = df_daily.ffill()
            
            df_clean = df_daily.reset_index()
            df_clean['subject_id'] = 1 

            df_long = pd.melt(
                df_clean, 
                id_vars=['subject_id', 'charttime'], 
                value_vars=available_tests,
                var_name='lab_test_name',
                value_name='valuenum'
            ).dropna()
            
            # Optimized Inference Extraction (milliseconds)
            X_extracted = extract_features(
                df_long,
                column_id='subject_id',
                column_sort='charttime',
                column_kind='lab_test_name',
                column_value='valuenum',
                kind_to_fc_parameters=kind_to_fc_parameters,
                impute_function=impute,
                disable_progressbar=True
            )
            
            if le is not None:
                try:
                    gender_encoded = le.transform([gender_val])[0]
                except Exception:
                    gender_encoded = 1 if gender_val == "M" else 0
            else:
                gender_encoded = 1 if gender_val == "M" else 0
                
            X_extracted['gender'] = gender_encoded
            X_extracted['age'] = age_val
            
            for col in selected_cols:
                if col not in X_extracted.columns:
                    X_extracted[col] = 0.0
            
            X_final = X_extracted[selected_cols]
            
            # -----------------------------------------------------------------------------
            # 5. Prediction & Clinical Explainability (Outputs)
            # -----------------------------------------------------------------------------
            
            # Calibrated probability
            prob = model.predict_proba(X_final)[0][1]
            
            st.divider()
            
            if prob < 0.20:
                risk_class = "risk-low"
                risk_text = "LOW RISK"
            elif prob < 0.50:
                risk_class = "risk-medium"
                risk_text = "MODERATE RISK"
            else:
                risk_class = "risk-high"
                risk_text = "HIGH RISK"
                
            st.markdown(f"""
                <div class="metric-container {risk_class}">
                    <div class="metric-title">CALIBRATED {risk_text}</div>
                    <div class="metric-value">{prob*100:.1f}%</div>
                    <div style="font-size: 16px;">Predicted Probability of 14-Day Hepatic Decompensation</div>
                </div>
            """, unsafe_allow_html=True)
            
            # SHAP XAI Engine using base XGBoost
            explainer = shap.TreeExplainer(base_xgb_model)
            shap_values = explainer(X_final)
            
            # Rename features for Clinical UI
            translated_names = [translate_feature_name(f) for f in shap_values.feature_names]
            shap_values.feature_names = translated_names
            
            # Clinical Translation Logic
            shap_vals = shap_values.values[0]
            feature_impact = {translated_names[i]: (shap_vals[i], abs(shap_vals[i])) for i in range(len(translated_names))}
            sorted_impact = sorted(feature_impact.items(), key=lambda x: x[1][1], reverse=True)
            
            top_features = []
            for feat, (val, abs_val) in sorted_impact[:3]:
                direction = "increasing" if val > 0 else "decreasing"
                top_features.append(f"the {direction} impact of **{feat}**")
                
            if len(top_features) > 1:
                summary_text = f"This {risk_text.lower()} profile is driven primarily by {', '.join(top_features[:-1])}, and {top_features[-1]}."
            elif len(top_features) == 1:
                summary_text = f"This {risk_text.lower()} profile is driven primarily by {top_features[0]}."
            else:
                summary_text = "Clinical trajectory indicates stable status."

            st.markdown("### AI Clinical Translation")
            st.markdown(f"""
                <div class="summary-box">
                    <strong>🩺 Auto-Summary:</strong> {summary_text}
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br><hr><br>", unsafe_allow_html=True)
            st.subheader("Deep Explainability (SHAP Waterfall Plot)")
            st.markdown("Visual breakdown of the calibrated top 30 longitudinal vectors shifting the risk probability.")
            
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        except Exception as e:
            st.error(f"❌ **Processing Error:** {e}")
            import traceback
            st.expander("Show Traceback").text(traceback.format_exc())
