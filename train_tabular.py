import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("  HepSense V2 - ML Tabular Engine Training")
    print("=" * 60)

    # 1. Load Training Matrix
    print("\n[1/6] Loading hepsense_training_matrix.csv...")
    if not os.path.exists('hepsense_training_matrix.csv'):
        print("[ERROR] hepsense_training_matrix.csv not found! Run data_pipeline.py first.")
        return

    data = pd.read_csv('hepsense_training_matrix.csv')
    print(f"  Loaded dataset with shape: {data.shape}")

    # Define targets and drop columns
    drop_cols = ['subject_id', 'index_hadm_id', 'index_admittime', 'index_dischtime', 'decompensation_90day', 'mortality_30day']
    feature_cols = [c for c in data.columns if c not in drop_cols]
    
    X = data[feature_cols]
    y = data['decompensation_90day'].astype(int)

    print(f"  Number of features: {len(feature_cols)}")
    print(f"  Positive decompensation events: {y.sum()} / {len(y)} ({y.mean()*100:.2f}%)")

    # 2. Handle Missing Values (Imputation)
    print("\n[2/6] Imputing missing values with column medians...")
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    
    # Save medians for inference scaling/imputation
    joblib.dump(medians, 'model_medians.pkl')
    print("  Saved medians to: model_medians.pkl")

    # 3. Train-Calibrate-Test Split
    print("\n[3/6] Splitting data (70% Train, 15% Calibration, 15% Test)...")
    # First split into train and temp (temp will be calib + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    # Split temp into calibration and test
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"  Train set size:       {X_train.shape[0]} (Pos: {y_train.sum()})")
    print(f"  Calibration set size: {X_calib.shape[0]} (Pos: {y_calib.sum()})")
    print(f"  Test set size:        {X_test.shape[0]} (Pos: {y_test.sum()})")

    # 4. Train base XGBoost model with class imbalance correction
    print("\n[4/6] Training base XGBoost Classifier with scale_pos_weight...")
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    spw = neg_count / pos_count
    print(f"  Configured scale_pos_weight: {spw:.2f}")

    # Set robust clinical classification hyperparameters
    base_xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )
    
    base_xgb.fit(X_train, y_train)

    # 5. Sigmoid Probability Calibration
    print("\n[5/6] Performing Sigmoid Calibration on calibration split...")
    from sklearn.frozen import FrozenEstimator
    
    calibrated_clf = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_xgb),
        method='sigmoid'
    )
    calibrated_clf.fit(X_calib, y_calib)

    # 6. Evaluate Model on Test Set
    print("\n[6/6] Evaluating final calibrated model on Test split...")
    y_probs = calibrated_clf.predict_proba(X_test)[:, 1]

    # Calculate ROC curve and Youden's J statistic to find the optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    j_scores = tpr - fpr
    best_index = np.argmax(j_scores)
    optimal_threshold = float(thresholds[best_index])
    
    print(f"  Optimal Clinical Threshold: {optimal_threshold:.4f}")

    # Apply the new threshold
    y_pred_optimal = (y_probs >= optimal_threshold).astype(int)

    test_auc = roc_auc_score(y_test, y_probs)
    test_ap = average_precision_score(y_test, y_probs)
    
    print("\n  --- Final Tabular Test Set Metrics ---")
    print(f"  ROC AUC:  {test_auc:.4f}")
    print(f"  PR AUC:   {test_ap:.4f}")
    print(f"  Threshold used: {optimal_threshold:.4f}")
    print("\n  Classification Report (Optimal Threshold):")
    print(classification_report(y_test, y_pred_optimal))

    # Save ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.axvline(x=fpr[best_index], color='gray', linestyle=':', label=f'Optimal J Cutoff')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {test_ap:.2f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig('tmeld_performance_curves.png', dpi=300)
    plt.close()
    print("  Saved performance curves -> tmeld_performance_curves.png")

    # 7. Generate SHAP Explainer
    print("\nGenerating SHAP explainer object...")
    explainer = shap.TreeExplainer(base_xgb)
    
    # Save the full model artifact package
    model_package = {
        'calibrated_model': calibrated_clf,
        'base_model': base_xgb,
        'medians': medians,
        'features': feature_cols,
        'optimal_threshold': optimal_threshold
    }
    
    joblib.dump(model_package, 'hepsense_temporal_xgb.joblib')
    joblib.dump(explainer, 'shap_explainer.joblib')
    
    print("\n[OK] Trained Model Package saved to: hepsense_temporal_xgb.joblib")
    print("[OK] SHAP Explainer saved to: shap_explainer.joblib")
    print("=" * 60)

if __name__ == "__main__":
    main()
