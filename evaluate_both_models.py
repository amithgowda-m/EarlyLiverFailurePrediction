import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)

ROOT = r"d:\EarlyLiverFailurePredictionN\EarlyLiverFailurePrediction"

def evaluate_ilpd():
    print("\n--- Evaluating Model 1: ILPD Liver Disease Classifier ---")
    data_path = os.path.join(ROOT, "ILPD", "indian_liver_patient.csv")
    model_path = os.path.join(ROOT, "ilpd_model_pkg.joblib")
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("[ERROR] ILPD dataset or model package not found.")
        return None
        
    df = pd.read_csv(data_path)
    df["target"] = (df["Dataset"] == 1).astype(int)
    df.drop(columns=["Dataset"], inplace=True)
    df["Gender"] = (df["Gender"] == "Male").astype(int)
    df["Albumin_and_Globulin_Ratio"].fillna(df["Albumin_and_Globulin_Ratio"].median(), inplace=True)
    
    FEATURES = [c for c in df.columns if c != "target"]
    X = df[FEATURES]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    pkg = joblib.load(model_path)
    model = pkg["model"]
    opt_thr = pkg["optimal_threshold"]
    
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
    prec, rec, pr_thresh = precision_recall_curve(y_test, y_prob)
    
    y_pred = (y_prob >= opt_thr).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC:  {pr_auc:.4f}")
    print(f"Youden J Threshold: {opt_thr:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Liver Patient"]))
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model 1: HepSense V3 ILPD Classifier Metrics", fontsize=14, fontweight="bold", y=1.02)
    
    # ROC
    axes[0].plot(fpr, tpr, color="#1e3a8a", lw=2.5, label=f"XGBoost (AUC = {roc_auc:.3f})")
    axes[0].plot([0,1],[0,1], "k--", lw=1.2, label="Random (AUC = 0.500)")
    axes[0].scatter(fpr[np.abs(roc_thresh - opt_thr).argmin()], tpr[np.abs(roc_thresh - opt_thr).argmin()], 
                    s=120, color="#be123c", zorder=5, label=f"Youden J @ {opt_thr:.2f}")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # PR
    no_skill = y_test.sum() / len(y_test)
    axes[1].plot(rec, prec, color="#059669", lw=2.5, label=f"XGBoost (PR AUC = {pr_auc:.3f})")
    axes[1].axhline(no_skill, color="gray", lw=1.2, linestyle="--", label=f"Baseline ({no_skill:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Confusion Matrix
    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Healthy", "Liver Pt"])
    axes[2].set_yticklabels(["Healthy", "Liver Pt"])
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")
    axes[2].set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(cm[i,j]), ha="center", va="center",
                         fontsize=18, fontweight="bold",
                         color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    
    plot_path = os.path.join(ROOT, "ilpd_performance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ILPD plot to {plot_path}")
    return roc_auc, pr_auc, opt_thr

def evaluate_temporal():
    print("\n--- Evaluating Model 2: MIMIC-IV Temporal Decompensation Classifier ---")
    data_path = os.path.join(ROOT, "hepsense_training_matrix.csv")
    model_path = os.path.join(ROOT, "hepsense_temporal_xgb.joblib")
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("[ERROR] Temporal dataset or model package not found.")
        return None
        
    data = pd.read_csv(data_path)
    drop_cols = ['subject_id', 'index_hadm_id', 'index_admittime', 'index_dischtime', 'decompensation_90day', 'mortality_30day']
    feature_cols = [c for c in data.columns if c not in drop_cols]
    
    X = data[feature_cols]
    y = data['decompensation_90day'].astype(int)
    
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    pkg = joblib.load(model_path)
    # The saved temporal model contains 'calibrated_model'
    model = pkg["calibrated_model"]
    opt_thr = pkg.get("optimal_threshold", 0.0366)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
    prec, rec, pr_thresh = precision_recall_curve(y_test, y_prob)
    
    y_pred = (y_prob >= opt_thr).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC:  {pr_auc:.4f}")
    print(f"Youden J Threshold: {opt_thr:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Stable", "Decompensating"]))
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model 2: HepSense V2 Temporal Classifier Metrics", fontsize=14, fontweight="bold", y=1.02)
    
    # ROC
    axes[0].plot(fpr, tpr, color="#1e3a8a", lw=2.5, label=f"XGBoost (AUC = {roc_auc:.3f})")
    axes[0].plot([0,1],[0,1], "k--", lw=1.2, label="Random (AUC = 0.500)")
    axes[0].scatter(fpr[np.abs(roc_thresh - opt_thr).argmin()], tpr[np.abs(roc_thresh - opt_thr).argmin()], 
                    s=120, color="#be123c", zorder=5, label=f"Youden J @ {opt_thr:.4f}")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # PR
    no_skill = y_test.sum() / len(y_test)
    axes[1].plot(rec, prec, color="#059669", lw=2.5, label=f"XGBoost (PR AUC = {pr_auc:.3f})")
    axes[1].axhline(no_skill, color="gray", lw=1.2, linestyle="--", label=f"Baseline ({no_skill:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Confusion Matrix
    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Stable", "Decomp"])
    axes[2].set_yticklabels(["Stable", "Decomp"])
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")
    axes[2].set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(cm[i,j]), ha="center", va="center",
                         fontsize=18, fontweight="bold",
                         color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    
    plot_path = os.path.join(ROOT, "hepsense_temporal_performance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Temporal plot to {plot_path}")
    return roc_auc, pr_auc, opt_thr

if __name__ == "__main__":
    evaluate_ilpd()
    evaluate_temporal()
