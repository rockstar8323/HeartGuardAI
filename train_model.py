"""
Heart Disease Prediction - Model Training Script
=================================================
Trains an ensemble VotingClassifier (Logistic Regression + Random Forest +
Gradient Boosting) on the UCI Heart Disease dataset for improved accuracy.
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
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'heart.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

FEATURE_LABELS = {
    'age': 'Age (years)',
    'sex': 'Sex (1=Male, 0=Female)',
    'cp': 'Chest Pain Type (0-3)',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Serum Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl',
    'restecg': 'Resting ECG Results (0-2)',
    'thalach': 'Max Heart Rate Achieved',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression (Exercise vs Rest)',
    'slope': 'Slope of Peak Exercise ST',
    'ca': 'Number of Major Vessels (0-3)',
    'thal': 'Thalassemia (0=Normal, 1=Fixed, 2=Reversible, 3=Other)'
}


def load_and_preprocess():
    """Load dataset, clean, and remove duplicates."""
    print("=" * 60)
    print("  HEART DISEASE PREDICTION - MODEL TRAINING (ENSEMBLE)")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"\n📊 Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    before = len(df)
    df = df.drop_duplicates()
    print(f"   Removed {before - len(df)} duplicate rows")
    print(f"   After cleaning: {df.shape[0]} samples")
    print(f"   Target distribution: {df['target'].value_counts().to_dict()}")

    X = df[FEATURE_NAMES].values

    # ── Target flip ──────────────────────────────────────────────────────
    # This Kaggle version of the UCI Cleveland dataset has the target
    # encoded as:  1 = No Disease (healthy),  0 = Disease (sick)
    # That is the OPPOSITE of the medical convention used everywhere
    # else in the codebase ("prediction==1 → High Risk").
    # Flip it here so the model learns: 1 = Disease, 0 = No Disease.
    y = 1 - df['target'].values

    #print(f"   Flipped target distribution (1=Disease): {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, df


def train_model(X, y):
    """Train ensemble VotingClassifier for improved accuracy."""
    print("\n🔧 Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=48, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")

    print("\n⚖️  Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define individual models
    print("\n🏗️  Building ensemble model...")

    lr = LogisticRegression(
        C=1, solver='lbfgs', max_iter=2000, 
        class_weight='balanced', random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced', random_state=42
    )

    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        min_samples_split=5, min_samples_leaf=2, random_state=42
    )

    # Cross-validate individual models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n📊 Cross-validation results (individual models):")
    for name, clf in [('Logistic Regression', lr), ('Random Forest', rf), ('Gradient Boosting', gb)]:
        scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        print(f"   {name:25s}: {scores.mean():.4f} (±{scores.std():.4f})")

    # Ensemble VotingClassifier (soft voting for probability-based)
    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[1, 1.5, 1.5]  # Give slightly more weight to tree-based models
    )

    print("\n🚀 Training VotingClassifier ensemble...")
    ensemble.fit(X_train_scaled, y_train)

    # Cross-validate ensemble
    ensemble_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"   Ensemble CV accuracy: {ensemble_scores.mean():.4f} (±{ensemble_scores.std():.4f})")

    return ensemble, scaler, X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate model and generate metrics."""
    print("\n📈 Model Evaluation (Test Set):")
    print("-" * 40)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.35).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"   Accuracy:  {acc:.4f}  ({acc*100:.1f}%)")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   AUC-ROC:   {auc:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   {cm}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics = {
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'auc_roc': round(float(auc), 4),
        'confusion_matrix': cm.tolist(),
        'roc_curve': {
            'fpr': [round(float(x), 6) for x in fpr],
            'tpr': [round(float(x), 6) for x in tpr]
        }
    }

    return metrics


def generate_shap_data(model, scaler, X_train_scaled):
    """Build per-estimator SHAP explainers (LinearExplainer for LR,
    TreeExplainer for RF and GB) and persist everything the web app
    needs to reconstruct them at startup without re-processing training
    data.

    Sub-models are saved as separate .pkl files so the Flask app can
    instantiate each explainer independently.  The LR background (a
    compact training-data sample) is stored in shap_data.json because
    LinearExplainer needs it; TreeExplainer uses the model's internal
    tree structure and requires no background dataset.

    Weights mirror the VotingClassifier: LR=1, RF=1.5, GB=1.5
    (total=4.0).  Combined expected value is the weighted mean of each
    individual explainer's expected value for class-1 (disease).
    """
    print("\n🧠 Generating SHAP explainer data (LinearExplainer + TreeExplainer × 2)...")

    # ── Extract the three fitted estimators from the ensemble ─────────
    lr = model.named_estimators_['lr']
    rf = model.named_estimators_['rf']
    gb = model.named_estimators_['gb']

    # ── Persist sub-models so the Flask app can load them at startup ──
    joblib.dump(lr, os.path.join(MODEL_DIR, 'shap_lr_model.pkl'))
    joblib.dump(rf, os.path.join(MODEL_DIR, 'shap_rf_model.pkl'))
    joblib.dump(gb, os.path.join(MODEL_DIR, 'shap_gb_model.pkl'))
    print("   Sub-models saved: shap_lr_model.pkl, shap_rf_model.pkl, shap_gb_model.pkl")

    # ── Background for LinearExplainer (100-sample subset of train data) ──
    lr_background = X_train_scaled[:100]

    # ── Build explainers ──────────────────────────────────────────────
    lr_explainer = shap.LinearExplainer(lr, lr_background)
    rf_explainer = shap.TreeExplainer(rf)
    gb_explainer = shap.TreeExplainer(gb)

    # ── Collect expected values (class-1 / disease probability axis) ──
    def _ev_class1(ev):
        """Return the class-1 scalar from a per-class or scalar expected value."""
        if hasattr(ev, '__len__') and len(ev) > 1:
            return float(ev[1])
        return float(ev if not hasattr(ev, '__len__') else ev[0])

    ev_lr = _ev_class1(lr_explainer.expected_value)
    ev_rf = _ev_class1(rf_explainer.expected_value)
    ev_gb = _ev_class1(gb_explainer.expected_value)

    # ── Weighted combination (mirrors VotingClassifier weights) ───────
    weights = [1.0, 1.5, 1.5]          # LR, RF, GB
    total_w = sum(weights)              # 4.0
    combined_ev = (weights[0] * ev_lr + weights[1] * ev_rf + weights[2] * ev_gb) / total_w

    shap_data = {
        'lr_background': lr_background.tolist(),    # needed by LinearExplainer
        'expected_value': round(combined_ev, 6),    # weighted combined EV (class-1)
        'weights': weights,                         # [LR, RF, GB]
        'total_weight': total_w,
        'individual_expected_values': {
            'lr': round(ev_lr, 6),
            'rf': round(ev_rf, 6),
            'gb': round(ev_gb, 6),
        }
    }

    print(f"   Expected values  — LR: {ev_lr:.4f}  RF: {ev_rf:.4f}  GB: {ev_gb:.4f}")
    print(f"   Combined EV (weighted, class-1): {combined_ev:.4f}")
    print("   ✅ SHAP data ready (LinearExplainer + TreeExplainer × 2)")

    return shap_data


def save_artifacts(model, scaler, metrics, shap_data):
    """Save model, scaler, metrics, SHAP data, and feature info.

    Note: the three SHAP sub-model .pkl files (shap_lr_model.pkl,
    shap_rf_model.pkl, shap_gb_model.pkl) are written by
    generate_shap_data() above.
    """
    print("\n💾 Saving artifacts...")

    joblib.dump(model, os.path.join(MODEL_DIR, 'heart_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODEL_DIR, 'shap_data.json'), 'w') as f:
        json.dump(shap_data, f)

    feature_info = {'names': FEATURE_NAMES, 'labels': FEATURE_LABELS}
    with open(os.path.join(MODEL_DIR, 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=2)

    print("   ✅ All artifacts saved to model/")


def main():
    X, y, df = load_and_preprocess()
    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test = train_model(X, y)
    metrics = evaluate_model(model, X_test_scaled, y_test)
    shap_data = generate_shap_data(model, scaler, X_train_scaled)
    save_artifacts(model, scaler, metrics, shap_data)

    print("\n" + "=" * 60)
    accuracy_pct = metrics['accuracy'] * 100
    print(f"  ✅ TRAINING COMPLETE — Accuracy: {accuracy_pct:.1f}%")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()