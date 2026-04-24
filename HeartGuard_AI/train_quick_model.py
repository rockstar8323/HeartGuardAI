"""
Quick Check Model — Heart Disease Prediction (Simplified)
==========================================================
Trains a model using only 7 features that a regular user
can easily know without needing medical tests.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import joblib

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'heart.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Features a user can easily know
QUICK_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'exang']

QUICK_LABELS = {
    'age': 'Age (years)',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Blood Pressure (mmHg)',
    'chol': 'Cholesterol (mg/dl)',
    'fbs': 'High Blood Sugar (>120 mg/dl)',
    'exang': 'Chest Pain During Exercise'
}


def main():
    print("=" * 60)
    print("  QUICK CHECK MODEL TRAINING")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_PATH)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().drop_duplicates()
    print(f"\n📊 Dataset: {df.shape[0]} samples")

    X = df[QUICK_FEATURES].values

    # ── Target flip ──────────────────────────────────────────────────────
    # This Kaggle version of the UCI Cleveland dataset uses:
    #   1 = No Disease (healthy),  0 = Disease (sick)
    # Flip so the model learns: 1 = Disease, 0 = No Disease
    y = 1 - df['target'].values

    print("\n🔧 Splitting data (fixed 80/20)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
        ],
        voting='soft', weights=[1, 1.5, 1.5]
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    y_prob = model.predict_proba(X_test_s)[:, 1]
    # y_pred = model.predict(X_test_s)
    y_pred = (y_prob >= 0.35).astype(int)

    metrics = {
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'auc_roc': round(float(roc_auc_score(y_test, y_prob)), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    print(f"\n📈 Test Results:")
    print(f"   Accuracy:  {metrics['accuracy']*100:.1f}%")
    print(f"   Precision: {metrics['precision']*100:.1f}%")
    print(f"   Recall:    {metrics['recall']*100:.1f}%")
    print(f"   F1-Score:  {metrics['f1_score']*100:.1f}%")
    print(f"   AUC-ROC:   {metrics['auc_roc']*100:.1f}%")

    # Save
    joblib.dump(model, os.path.join(MODEL_DIR, 'quick_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'quick_scaler.pkl'))

    # Save LR sub-model for SHAP
    lr_model = model.named_estimators_['lr']
    joblib.dump(lr_model, os.path.join(MODEL_DIR, 'quick_lr_model.pkl'))

    with open(os.path.join(MODEL_DIR, 'quick_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    quick_info = {'names': QUICK_FEATURES, 'labels': QUICK_LABELS}
    with open(os.path.join(MODEL_DIR, 'quick_feature_info.json'), 'w') as f:
        json.dump(quick_info, f, indent=2)

    # Save SHAP background
    import shap
    shap_bg = X_train_s[:80].tolist()
    explainer = shap.LinearExplainer(lr_model, X_train_s[:80])
    shap_data = {
        'background': shap_bg,
        'expected_value': float(explainer.expected_value)
    }
    with open(os.path.join(MODEL_DIR, 'quick_shap_data.json'), 'w') as f:
        json.dump(shap_data, f)

    print(f"\n✅ Quick Check model saved! Accuracy: {metrics['accuracy']*100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()