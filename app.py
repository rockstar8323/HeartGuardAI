"""
Heart Disease Prediction - Flask Web Application
=================================================
Full-stack app with user authentication, dual-mode ML prediction
(Quick Check + Detailed Clinical with report upload), SHAP
explanations, and prediction history storage.
"""

import os
import json
import base64
import io
import warnings
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, PredictionHistory
from report_parser import parse_report

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'heartguard-ai-secret-key-2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'heartguard.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access the prediction tool.'
login_manager.login_message_category = 'info'

with app.app_context():
    db.create_all()

# ─── Load Detailed Model (Ensemble — 13 features) ───────────────────
print("Loading model artifacts...")
model = joblib.load(os.path.join(MODEL_DIR, 'heart_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

with open(os.path.join(MODEL_DIR, 'metrics.json'), 'r') as f:
    metrics = json.load(f)
with open(os.path.join(MODEL_DIR, 'shap_data.json'), 'r') as f:
    shap_data = json.load(f)
with open(os.path.join(MODEL_DIR, 'feature_info.json'), 'r') as f:
    feature_info = json.load(f)

FEATURE_NAMES = feature_info['names']
FEATURE_LABELS = feature_info['labels']

# ── Per-estimator SHAP explainers — built once at startup ────────────
# Weights mirror VotingClassifier: LR=1, RF=1.5, GB=1.5  (total 4.0)
_shap_lr_model = joblib.load(os.path.join(MODEL_DIR, 'shap_lr_model.pkl'))
_shap_rf_model = joblib.load(os.path.join(MODEL_DIR, 'shap_rf_model.pkl'))
_shap_gb_model = joblib.load(os.path.join(MODEL_DIR, 'shap_gb_model.pkl'))

_lr_background = np.array(shap_data['lr_background'])   # needed by LinearExplainer
SHAP_WEIGHTS   = shap_data['weights']                   # [1.0, 1.5, 1.5]
SHAP_TOTAL_W   = shap_data['total_weight']              # 4.0

lr_explainer = shap.LinearExplainer(_shap_lr_model, _lr_background)
rf_explainer = shap.TreeExplainer(_shap_rf_model)
gb_explainer = shap.TreeExplainer(_shap_gb_model)

print(f"   [OK] Detailed model - Accuracy: {metrics['accuracy']*100:.1f}%")

# ─── Load Quick Model (7 features) ──────────────────────────────────
quick_model = joblib.load(os.path.join(MODEL_DIR, 'quick_model.pkl'))
quick_scaler = joblib.load(os.path.join(MODEL_DIR, 'quick_scaler.pkl'))
quick_lr = joblib.load(os.path.join(MODEL_DIR, 'quick_lr_model.pkl'))

with open(os.path.join(MODEL_DIR, 'quick_metrics.json'), 'r') as f:
    quick_metrics = json.load(f)
with open(os.path.join(MODEL_DIR, 'quick_feature_info.json'), 'r') as f:
    quick_info = json.load(f)
with open(os.path.join(MODEL_DIR, 'quick_shap_data.json'), 'r') as f:
    quick_shap_data = json.load(f)

quick_bg = np.array(quick_shap_data['background'])
quick_explainer = shap.LinearExplainer(quick_lr, quick_bg)
QUICK_FEATURES = quick_info['names']
QUICK_LABELS = quick_info['labels']

print(f"   [OK] Quick model - Accuracy: {quick_metrics['accuracy']*100:.1f}%")
print("[OK] All models loaded!")


# ─── Flask-Login ─────────────────────────────────────────────────────
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ─── Auth Routes ─────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=True)
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(request.args.get('next') or url_for('index'))
        else:
            flash('Invalid email or password.', 'error')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        if not name or not email or not password:
            flash('All fields are required.', 'error')
        elif password != confirm:
            flash('Passwords do not match.', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
        elif User.query.filter_by(email=email).first():
            flash('An account with this email already exists.', 'error')
        else:
            user = User(name=name, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user, remember=True)
            flash(f'Welcome to HeartGuard AI, {name}!', 'success')
            return redirect(url_for('index'))
    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


# ─── Main Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', metrics=metrics)


@app.route('/predict', methods=['GET'])
@login_required
def predict_form():
    return render_template('predict.html', features=FEATURE_LABELS)


@app.route('/predict/quick', methods=['POST'])
@login_required
def predict_quick():
    """Quick Check — 7 user-friendly features."""
    try:
        input_values = []
        for feat in QUICK_FEATURES:
            val = float(request.form.get(feat, 0))
            input_values.append(val)

        X_input = np.array(input_values).reshape(1, -1)
        X_scaled = quick_scaler.transform(X_input)

        THRESHOLD = 0.35
        probability = quick_model.predict_proba(X_scaled)[0]
        prediction = 1 if probability[1] >= THRESHOLD else 0
        confidence = abs(probability[0] - probability[1]) * 100

        # SHAP values — positive = increases disease risk, negative = decreases
        shap_vals = quick_explainer.shap_values(X_scaled)
        shap_vals_1d = shap_vals[0]  # shape (n_features,)
        
        contributions = []
        for i, feat in enumerate(QUICK_FEATURES):
            contributions.append({
                'feature': QUICK_LABELS[feat],
                'feature_key': feat,
                'value': float(input_values[i]),
                'shap_value': float(shap_vals_1d[i]),
                'impact': 'increases' if shap_vals_1d[i] > 0 else 'decreases'
            })
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        shap_plot_b64 = generate_shap_plot(shap_vals_1d, input_values, QUICK_FEATURES, QUICK_LABELS)

        result = {
            'prediction': int(prediction),
            'prediction_label': 'High Risk of Heart Disease' if prediction == 1 else 'Low Risk of Heart Disease',
            'confidence': round(confidence, 1),
            'probability_disease': round(float(probability[1]) * 100, 1),
            'probability_no_disease': round(float(probability[0]) * 100, 1),
            'contributions': contributions,
            'shap_plot': shap_plot_b64,
            'input_values': dict(zip(QUICK_FEATURES, input_values)),
            'mode': 'Quick Check',
            'mode_note': 'This is a preliminary assessment using basic health information. For a more detailed analysis, use Detailed Clinical mode with all 13 parameters.'
        }

        # Save to database (quick mode — only 7 features)
        history = PredictionHistory(
            user_id=current_user.id,
            age=input_values[0], sex=input_values[1], cp=input_values[2],
            trestbps=input_values[3], chol=input_values[4], fbs=input_values[5],
            exang=input_values[6],
            prediction=int(prediction),
            confidence=round(confidence, 1),
            probability_disease=round(float(probability[1]) * 100, 1),
            probability_no_disease=round(float(probability[0]) * 100, 1)
        )
        db.session.add(history)
        db.session.commit()

        return render_template('result.html', result=result, metrics=metrics)

    except Exception as e:
        return render_template('predict.html', features=FEATURE_LABELS,
                               error=f"Prediction failed: {str(e)}")


@app.route('/predict/detailed', methods=['POST'])
@login_required
def predict_detailed():
    """Detailed Clinical — all 13 features."""
    try:
        input_values = []
        for feat in FEATURE_NAMES:
            val = float(request.form.get(feat, 0))
            input_values.append(val)

        X_input = np.array(input_values).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        THRESHOLD = 0.35
        probability = model.predict_proba(X_scaled)[0]
        prediction = 1 if probability[1] >= THRESHOLD else 0
        confidence = abs(probability[0] - probability[1]) * 100

        # ── SHAP: weighted combination of per-estimator explanations ─────
        # LinearExplainer → shape (1, n); pick row 0 → (n,)
        lr_shap = lr_explainer.shap_values(X_scaled)[0]

        # TreeExplainer output varies by SHAP version / estimator:
        #   list  [shape(1,n), shape(1,n)] — older SHAP, per-class list
        #   ndarray shape (1, n, 2)        — newer SHAP, last axis = class
        #   ndarray shape (1, n)           — rare single-output fallback
        # Always extract a flat (n,) array for class-1 (disease).
        def _class1_shap(raw):
            if isinstance(raw, list):
                arr = raw[1] if len(raw) > 1 else raw[0]
                return arr[0]           # (1, n) → (n,)
            if raw.ndim == 3:           # (1, n_features, n_classes)
                return raw[0, :, 1]     # → (n,), class-1
            return raw[0]               # (1, n) → (n,)

        rf_shap = _class1_shap(rf_explainer.shap_values(X_scaled))
        gb_shap = _class1_shap(gb_explainer.shap_values(X_scaled))

        # Weighted average mirrors VotingClassifier weights [LR=1, RF=1.5, GB=1.5]
        shap_vals_disease = (
            SHAP_WEIGHTS[0] * lr_shap +
            SHAP_WEIGHTS[1] * rf_shap +
            SHAP_WEIGHTS[2] * gb_shap
        ) / SHAP_TOTAL_W
        
        contributions = []
        for i, feat in enumerate(FEATURE_NAMES):
            contributions.append({
                'feature': FEATURE_LABELS[feat],
                'feature_key': feat,
                'value': float(input_values[i]),
                'shap_value': float(shap_vals_disease[i]),
                'impact': 'increases' if shap_vals_disease[i] > 0 else 'decreases'
            })
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        shap_plot_b64 = generate_shap_plot(shap_vals_disease, input_values, FEATURE_NAMES, FEATURE_LABELS)

        result = {
            'prediction': int(prediction),
            'prediction_label': 'High Risk of Heart Disease' if prediction == 1 else 'Low Risk of Heart Disease',
            'confidence': round(confidence, 1),
            'probability_disease': round(float(probability[1]) * 100, 1),
            'probability_no_disease': round(float(probability[0]) * 100, 1),
            'contributions': contributions,
            'shap_plot': shap_plot_b64,
            'input_values': dict(zip(FEATURE_NAMES, input_values)),
            'mode': 'Detailed Clinical'
        }

        history = PredictionHistory(
            user_id=current_user.id,
            age=input_values[0], sex=input_values[1], cp=input_values[2],
            trestbps=input_values[3], chol=input_values[4], fbs=input_values[5],
            restecg=input_values[6], thalach=input_values[7], exang=input_values[8],
            oldpeak=input_values[9], slope=input_values[10], ca=input_values[11],
            thal=input_values[12],
            prediction=int(prediction),
            confidence=round(confidence, 1),
            probability_disease=round(float(probability[1]) * 100, 1),
            probability_no_disease=round(float(probability[0]) * 100, 1)
        )
        db.session.add(history)
        db.session.commit()

        return render_template('result.html', result=result, metrics=metrics)

    except Exception as e:
        return render_template('predict.html', features=FEATURE_LABELS,
                               error=f"Prediction failed: {str(e)}")


@app.route('/predict/upload', methods=['POST'])
@login_required
def predict_upload():
    """Accept file upload → extract medical values → return JSON."""
    if 'report' not in request.files:
        return jsonify({'error': 'No file uploaded', 'extracted': {}, 'fields_found': 0})

    file = request.files['report']
    if file.filename == '':
        return jsonify({'error': 'No file selected', 'extracted': {}, 'fields_found': 0})

    file_bytes = file.read()
    result = parse_report(file_bytes, file.filename)
    return jsonify(result)


@app.route('/dashboard')
@login_required
def dashboard():
    predictions = current_user.predictions.all()
    return render_template('dashboard.html', predictions=predictions, metrics=metrics)


@app.route('/model-performance')
def model_performance():
    return render_template('performance.html', metrics=metrics)


@app.route('/api/metrics')
def api_metrics():
    return jsonify(metrics)


def generate_shap_plot(shap_values_single, feature_values, feat_names, feat_labels):
    """Generate a horizontal bar chart of SHAP values."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(feat_names) * 0.5)))

    indices = np.argsort(np.abs(shap_values_single))
    sorted_features = [feat_labels[feat_names[i]] for i in indices]
    sorted_shap = [shap_values_single[i] for i in indices]
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in sorted_shap]

    ax.barh(range(len(sorted_features)), sorted_shap, color=colors, height=0.6, edgecolor='none')
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=10)
    ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Impact on Heart Disease Prediction', fontsize=14, fontweight='bold', pad=15)
    ax.axvline(x=0, color='#555', linewidth=0.8, linestyle='-')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#ccc')
    ax.spines['left'].set_color('#ccc')
    ax.tick_params(colors='#555')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Increases Risk'),
        Patch(facecolor='#2ecc71', label='Decreases Risk')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
