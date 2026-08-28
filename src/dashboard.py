
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import re
import matplotlib.pyplot as plt
import shap
import requests
import warnings
from datetime import datetime
import io

warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
def load_css():
    st.markdown("""
    <style>
        :root {
            --primary: #3B82C4;
            --primary-dark: #2C6B9E;
            --primary-light: #EAF3FC;
            --danger: #E5484D;
            --danger-bg: #FDECEC;
            --success: #2F9E62;
            --success-bg: #EAF7EF;
            --warning: #D97706;
            --text: #16202B;
            --text-secondary: #45566B;
            --text-muted: #7C8AA0;
            --border: #E3EAF2;
        }

        .main { background-color: #F6F9FC; }

        .welcome-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FBFF 100%);
            border-radius: 18px;
            padding: 2.4rem 2.8rem;
            max-width: 780px;
            margin: 1rem auto;
            box-shadow: 0 6px 24px rgba(44,107,158,0.10);
            border: 1px solid var(--border);
            text-align: center;
        }
        .welcome-card h1 { font-size: 2.1rem; margin-bottom: 0.2rem; color: var(--text); }
        .welcome-card .subtitle { color: var(--text-secondary); font-size: 1.05rem; margin-bottom: 1.2rem; }
        .welcome-card .stat-row { display: flex; justify-content: center; gap: 2.2rem; margin: 1rem 0; flex-wrap: wrap; }
        .welcome-card .stat { text-align: center; }
        .welcome-card .stat .num { font-size: 1.5rem; font-weight: 700; color: var(--primary-dark); }
        .welcome-card .stat .lbl { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.4px; }

        .welcome-section {
            max-width: 780px;
            margin: 1.4rem auto;
            text-align: center;
        }
        .welcome-section h3 {
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            color: var(--text);
        }
        .welcome-section p, .welcome-section li { color: var(--text); font-size: 0.95rem; line-height: 1.6; text-align: center; }
        .welcome-section ul { display: inline-block; text-align: left; list-style-position: outside; padding-left: 1.2rem; margin: 0.6rem auto; }

        .model-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin: 0 auto;
        }
        .model-table th { background: var(--primary-light); text-align: center; padding: 0.6rem 1rem; font-weight: 600; color: var(--text); }
        .model-table td { padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); color: var(--text); text-align: center; }
        .model-table .selected { background: #EBF5FF; font-weight: 600; }
        .model-table .selected td:first-child::before { content: "✅ "; }
        .model-note { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.6rem; text-align: center; }

        .welcome-btn-row { max-width: 780px; margin: 0 auto; }

        .header-accent {
            height: 5px;
            width: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 60%, #6FB1E8 100%);
            margin-bottom: 1.1rem;
        }
        .section-title { text-align: center; color: var(--text); font-weight: 700; margin: 0.4rem 0 0.9rem 0; }

        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            border: 1px solid var(--border);
            border-left: 5px solid var(--primary);
            box-shadow: 0 1px 6px rgba(0,0,0,0.05);
            transition: box-shadow 0.2s, transform 0.2s;
        }
        .metric-card:hover { box-shadow: 0 6px 18px rgba(59,130,196,0.18); transform: translateY(-2px); }
        .metric-card.accent-danger { border-left-color: var(--danger); }
        .metric-card.accent-success { border-left-color: var(--success); }
        .metric-card .label { color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.4px; font-weight: 600; }
        .metric-card .value { color: var(--text); font-size: 1.6rem; font-weight: 700; margin: 0.1rem 0; }
        .metric-card .sub { color: var(--text-muted); font-size: 0.85rem; }

        .result-banner { border-radius: 12px; padding: 1.1rem 1.6rem; margin: 0.5rem 0; border: 1px solid; }
        .result-banner.attack { background: var(--danger-bg); border-color: var(--danger); }
        .result-banner.benign { background: var(--success-bg); border-color: var(--success); }
        .result-banner .title { font-size: 1.25rem; font-weight: 700; }
        .result-banner.attack .title { color: var(--danger); }
        .result-banner.benign .title { color: var(--success); }

        .ai-box { padding: 0.8rem 1rem; border-radius: 8px; margin: 0.4rem 0; border-left: 3px solid var(--primary); background: #FAFCFE; }
        .ai-box .heading { font-weight: 600; font-size: 0.9rem; color: var(--text-secondary); }

        .log-entry { padding: 0.3rem 0; border-bottom: 1px solid #F0F2F5; font-size: 0.9rem; }
        .log-badge { padding: 0.1rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
        .log-badge.attack { background: var(--danger-bg); color: var(--danger); }
        .log-badge.benign { background: var(--success-bg); color: var(--success); }

        .stButton > button {
            border-radius: 9px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            transition: all 0.2s !important;
            color: var(--text) !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
            color: white !important;
            border: none !important;
        }
        .stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 16px rgba(59,130,196,0.3) !important; }

        .stTabs [data-baseweb="tab"] { font-size: 1rem !important; font-weight: 600 !important; color: var(--text) !important; }
        .stTabs [data-baseweb="tab"]:hover { color: var(--primary-dark) !important; }
        .stTabs [aria-selected="true"] { color: var(--primary-dark) !important; border-bottom-color: var(--primary) !important; }

        .footer { text-align: center; padding: 1.5rem 0 0.5rem 0; color: var(--text-muted); font-size: 0.85rem; border-top: 1px solid var(--border); margin-top: 2rem; }

        h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
        .stMarkdown p, .stMarkdown li, .stMarkdown div { color: var(--text); }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ============================================
# SESSION STATE
# ============================================
if 'welcome_dismissed' not in st.session_state:
    st.session_state.welcome_dismissed = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'ai_summaries' not in st.session_state:
    st.session_state.ai_summaries = []
if 'latest' not in st.session_state:
    st.session_state.latest = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'sample_test_options' not in st.session_state:
    st.session_state.sample_test_options = None
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'X_full' not in st.session_state:
    st.session_state.X_full = None
if 'y_full' not in st.session_state:
    st.session_state.y_full = None
if 'attack_labels' not in st.session_state:
    st.session_state.attack_labels = None
if 'test_indices' not in st.session_state:
    st.session_state.test_indices = None
if 'show_shap' not in st.session_state:
    st.session_state.show_shap = False
if 'show_ai_summary' not in st.session_state:
    st.session_state.show_ai_summary = False
if 'cached_shap_fig' not in st.session_state:
    st.session_state.cached_shap_fig = None
if 'cached_ai_summary' not in st.session_state:
    st.session_state.cached_ai_summary = None

# ============================================
# PERSISTENT LOG
# ============================================
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "ids_logs.csv")
os.makedirs(LOG_DIR, exist_ok=True)

if 'log_df' not in st.session_state:
    if os.path.exists(LOG_FILE):
        try:
            st.session_state.log_df = pd.read_csv(LOG_FILE)
        except:
            st.session_state.log_df = pd.DataFrame(columns=[
                'Timestamp', 'Source', 'Prediction', 'Confidence',
                'Ground_Truth', 'MITRE', 'Correct', 'Top_Features'
            ])
    else:
        st.session_state.log_df = pd.DataFrame(columns=[
            'Timestamp', 'Source', 'Prediction', 'Confidence',
            'Ground_Truth', 'MITRE', 'Correct', 'Top_Features'
        ])

# ============================================
# HELPER FUNCTIONS
# ============================================
def clean_attack_type(name):
    if name is None:
        return None
    replacements = {'\ufffd': '', '\u2013': '-', '\u2014': '-', '"': '', "'": ''}
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.strip()

def get_mitre_from_attack_type(attack_type):
    mapping = {
        'DoS Hulk': 'T1499',
        'DoS GoldenEye': 'T1499',
        'DoS slowloris': 'T1499',
        'DoS Slowhttptest': 'T1499',
        'DDoS': 'T1498',
        'PortScan': 'T1046',
        'FTP-Patator': 'T1110',
        'SSH-Patator': 'T1110',
        'Web Attack Brute Force': 'T1190',
        'Web Attack XSS': 'T1190',
        'Web Attack Sql Injection': 'T1190',
        'Bot': 'T1071',
        'Infiltration': 'T1071',
        'Heartbleed': 'T1190',
        'BENIGN': None,
    }
    if attack_type:
        clean = clean_attack_type(attack_type)
        return mapping.get(clean, 'T1499')
    return 'T1499'

def save_log(entry):
    new_row = pd.DataFrame([entry])
    if len(st.session_state.log_df) == 0:
        st.session_state.log_df = new_row
    else:
        st.session_state.log_df = pd.concat([st.session_state.log_df, new_row], ignore_index=True)
    try:
        st.session_state.log_df.to_csv(LOG_FILE, index=False)
    except:
        pass

# ============================================
# LOAD MODELS & DATA
# ============================================
@st.cache_resource
def load_models_and_data():
    status = st.empty()
    status.info("🔄 Loading models and data...")

    with open('models/random_forest_multiday.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    X_full = pd.read_csv('data/processed/X_processed_multiday.csv')
    y_full = pd.read_csv('data/processed/y_processed_multiday.csv').values.ravel()
    attack_df = pd.read_csv('data/processed/attack_types_multiday.csv')
    attack_labels = attack_df['attack_type'].apply(clean_attack_type).values
    test_indices = pd.read_csv('data/processed/test_indices.csv').values.ravel()

    feature_names = X_full.columns.tolist()

    status.success("✅ Ready")
    time.sleep(0.3)
    status.empty()

    return rf_model, feature_names, X_full, y_full, attack_labels, test_indices

def get_shap_explainer(rf_model):
    if st.session_state.explainer is None:
        with st.spinner("⏳ Loading SHAP..."):
            st.session_state.explainer = shap.TreeExplainer(rf_model)
    return st.session_state.explainer

# ============================================
# LOAD DATA
# ============================================
try:
    rf_model, feature_names, X_full, y_full, attack_labels, test_indices = load_models_and_data()
    st.session_state.rf_model = rf_model
    st.session_state.feature_names = feature_names
    st.session_state.X_full = X_full
    st.session_state.y_full = y_full
    st.session_state.attack_labels = attack_labels
    st.session_state.test_indices = test_indices
    st.session_state.models_loaded = True
except Exception as e:
    st.session_state.models_loaded = False
    st.error(f"❌ Failed to load: {e}")

# ============================================
# BUILD TEST OPTIONS
# ============================================
def build_test_options():
    options = []
    for t in pd.unique(attack_labels):
        idx_for_type = test_indices[attack_labels[test_indices] == t]
        chosen = np.random.choice(idx_for_type, min(3, len(idx_for_type)), replace=False)
        for idx in chosen:
            label = f"Row {idx} — {t if t != 'BENIGN' else 'Benign'}"
            options.append((int(idx), label))
    return sorted(options, key=lambda x: x[1])

if st.session_state.models_loaded and st.session_state.sample_test_options is None:
    st.session_state.sample_test_options = build_test_options()

# ============================================
# ANALYZE SAMPLE
# ============================================
def analyze_sample(X_row, source_label, attack_type=None):
    rf_model = st.session_state.rf_model
    feature_names = st.session_state.feature_names

    pred = rf_model.predict(X_row)[0]
    proba = rf_model.predict_proba(X_row)[0]

    shap_explainer = get_shap_explainer(rf_model)
    shap_values = shap_explainer.shap_values(X_row)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    shap_imp = np.abs(shap_values).flatten()
    top_idx = np.argsort(shap_imp)[-5:][::-1]

    top_features = []
    for idx in top_idx:
        if shap_imp[idx] > 0.01:
            top_features.append({
                'feature': feature_names[idx],
                'value': float(X_row[0, idx]),
                'shap': float(shap_values[0, idx]),
                'direction': 'attack' if shap_values[0, idx] > 0 else 'benign'
            })

    clean_attack = clean_attack_type(attack_type)
    mitre = get_mitre_from_attack_type(clean_attack)

    is_correct = False
    if pred == 1 and clean_attack and clean_attack != 'BENIGN':
        is_correct = True
    elif pred == 0 and clean_attack == 'BENIGN':
        is_correct = True
    elif pred == 0 and clean_attack and clean_attack != 'BENIGN':
        is_correct = False
    elif pred == 1 and clean_attack == 'BENIGN':
        is_correct = False

    confidence = float(proba[pred])

    result = {
        'timestamp': time.strftime('%H:%M:%S'),
        'source': source_label,
        'prediction': int(pred),
        'confidence': confidence,
        'mitre': mitre,
        'top_features': top_features[:5],
        'shap_values': shap_values,
        'X_row': X_row,
        'attack_type': clean_attack if clean_attack else 'Unknown',
        'is_correct': is_correct,
    }

    st.session_state.history.insert(0, result)

    st.session_state.show_shap = False
    st.session_state.show_ai_summary = False
    st.session_state.cached_shap_fig = None
    st.session_state.cached_ai_summary = None

    if pred == 1:
        st.session_state.alerts.append(result)
        ai_summary = get_ollama_summary(pred, confidence, mitre, top_features, clean_attack)
        st.session_state.ai_summaries.append(ai_summary)
        result['ai_summary'] = ai_summary

    log_entry = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Source': source_label,
        'Prediction': 'Attack' if pred == 1 else 'Benign',
        'Confidence': f"{confidence*100:.1f}%",
        'Ground_Truth': clean_attack if clean_attack else 'Unknown',
        'MITRE': mitre if mitre else 'N/A',
        'Correct': 'Yes' if is_correct else 'No',
        'Top_Features': ', '.join([f"{f['feature']}: {f['value']:.2f}" for f in top_features[:3]])
    }
    save_log(log_entry)

    return result

# ============================================
# GET OLLAMA SUMMARY
# ============================================
def get_ollama_summary(pred, confidence, mitre, top_features, attack_type=None):
    feature_list = '\n'.join([f"- {f['feature']}: {f['value']:.2f} (SHAP: {f['shap']:.3f})" for f in top_features[:5]])
    attack_note = f"\nGround Truth: {attack_type}" if attack_type else ""

    prompt = f"""You are a senior SOC analyst explaining a security detection.

ALERT DETAILS:
- Classification: {'ATTACK' if pred == 1 else 'BENIGN'}
- Confidence: {confidence:.2%}
- MITRE Technique: {mitre}
- Top Features: {feature_list}
{attack_note}

REQUIRED OUTPUT — Write in clear, plain English:

**WHAT HAPPENED:** (1-2 sentences)

**IMPACT:** (1 sentence)

**RECOMMENDED ACTION:** (2-3 steps)

**CONTEXT:** (1 sentence)

Keep it under 200 words.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 400, "temperature": 0.7}
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get('response', 'No response from AI')
        else:
            return f"❌ Ollama error: {response.status_code}"
    except:
        return "❌ Could not connect to Ollama."

# ============================================
# PARSE AI SUMMARY
# ============================================
def parse_ai_summary(text):
    sections = {'what_happened': '', 'impact': '', 'action': '', 'context': '', 'risk': 'Medium'}
    text = text.replace('*', '').strip()

    patterns = {
        'what_happened': r'WHAT HAPPENED\**\s*[:]?\s*(.*?)(?=\n?\s*(?:IMPACT|WHY|WHAT|CONTEXT|$))',
        'impact': r'IMPACT\**\s*[:]?\s*(.*?)(?=\n?\s*(?:WHAT SHOULD|RECOMMENDED|CONTEXT|$))',
        'action': r'(?:WHAT SHOULD I DO|RECOMMENDED ACTION)\**\s*[:]?\s*(.*?)(?=\n?\s*(?:CONTEXT|$))',
        'context': r'CONTEXT\**\s*[:]?\s*(.*?)(?=$)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            cleaned = match.group(1).strip().strip('*').strip()
            if cleaned and len(cleaned) > 10:
                sections[key] = cleaned

    if 'critical' in text.lower():
        sections['risk'] = 'Critical'
    elif 'high' in text.lower():
        sections['risk'] = 'High'
    elif 'medium' in text.lower():
        sections['risk'] = 'Medium'
    elif 'low' in text.lower():
        sections['risk'] = 'Low'

    return sections

# ============================================
# GENERATE SHAP PLOT
# ============================================
def generate_shap_plot(X_row, shap_values):
    try:
        rf_model = st.session_state.rf_model
        shap_explainer = get_shap_explainer(rf_model)
        base_value = shap_explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=base_value,
                data=X_row[0],
                feature_names=st.session_state.feature_names
            ),
            show=False
        )
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

# ============================================
# RENDER RESULT
# ============================================
def render_result(result):
    pred = result['prediction']
    confidence = result['confidence']
    mitre = result.get('mitre', '')
    attack_type = result.get('attack_type', 'Unknown')
    is_correct = result.get('is_correct', False)
    top_features = result.get('top_features', [])
    shap_values = result.get('shap_values', None)
    X_row = result.get('X_row', None)

    if pred == 1:
        st.markdown(f"""
        <div class="result-banner attack">
            <div style="display: flex; align-items: center; gap: 0.8rem;">
                <span style="font-size: 1.8rem;">🚨</span>
                <div>
                    <div class="title">Attack Detected</div>
                    <div style="color: #45566B; font-size: 0.95rem;">Confidence: {confidence*100:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-banner benign">
            <div style="display: flex; align-items: center; gap: 0.8rem;">
                <span style="font-size: 1.8rem;">✅</span>
                <div>
                    <div class="title">Benign Traffic</div>
                    <div style="color: #45566B; font-size: 0.95rem;">Confidence: {confidence*100:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h3 class="section-title">Model vs Ground Truth</h3>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Model:** {'Attack' if pred == 1 else 'Benign'}")
    with c2:
        st.markdown(f"**Ground Truth:** {attack_type if attack_type else 'Unknown'}")
    with c3:
        if attack_type and attack_type != 'Unknown':
            if is_correct:
                st.success("✅ Correct")
            else:
                st.error("❌ Incorrect")
        else:
            st.caption("—")

    if mitre:
        st.caption(f"🎯 MITRE: {mitre}")

    st.markdown("---")

    m1, m2, m3 = st.columns(3)
    accent = "accent-danger" if pred == 1 else "accent-success"
    with m1:
        st.markdown(f'<div class="metric-card {accent}"><div class="label">Confidence</div><div class="value">{confidence*100:.1f}%</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card {accent}"><div class="label">Risk Level</div><div class="value">{"HIGH" if pred == 1 else "LOW"}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card {accent}"><div class="label">MITRE</div><div class="value">{mitre if mitre else "N/A"}</div></div>', unsafe_allow_html=True)

    st.markdown("###  ")
    st.subheader("Top Contributing Features")
    if top_features:
        df = pd.DataFrame(top_features)
        df.columns = ['Feature', 'Value', 'SHAP', 'Direction']
        st.dataframe(df, use_container_width=True)

        chart_df = pd.DataFrame({
            'feature': [f['feature'] for f in top_features],
            'SHAP': [f['shap'] for f in top_features]
        }).set_index('feature')
        st.bar_chart(chart_df)

    if X_row is not None and shap_values is not None:
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📊 Show SHAP", key="show_shap_btn"):
                st.session_state.show_shap = not st.session_state.show_shap
                if st.session_state.show_shap:
                    with st.spinner("Generating SHAP plot..."):
                        fig = generate_shap_plot(X_row, shap_values)
                        st.session_state.cached_shap_fig = fig
                else:
                    st.session_state.cached_shap_fig = None
                st.rerun()
        with col2:
            st.caption("Click to view SHAP waterfall plot explaining this prediction")

        if st.session_state.show_shap and st.session_state.cached_shap_fig is not None:
            st.pyplot(st.session_state.cached_shap_fig)
            plt.close(st.session_state.cached_shap_fig)

    if pred == 1 and 'ai_summary' in result:
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🤖 Show AI Summary", key="show_ai_btn"):
                st.session_state.show_ai_summary = not st.session_state.show_ai_summary
                if st.session_state.show_ai_summary:
                    st.session_state.cached_ai_summary = result['ai_summary']
                else:
                    st.session_state.cached_ai_summary = None
                st.rerun()
        with col2:
            st.caption("Click to view AI-generated triage summary")

        if st.session_state.show_ai_summary and st.session_state.cached_ai_summary:
            render_ai_summary(st.session_state.cached_ai_summary)

    with st.expander("View Raw Data"):
        st.dataframe(X_row, use_container_width=True)

# ============================================
# RENDER AI SUMMARY
# ============================================
def render_ai_summary(text):
    parsed = parse_ai_summary(text)

    st.markdown("---")
    st.subheader("🤖 AI Triage Summary")

    risk = parsed.get('risk', 'Medium')
    if risk in ['Critical', 'High']:
        st.error(f"**Risk Level: {risk}**")
    elif risk == 'Medium':
        st.warning(f"**Risk Level: {risk}**")
    else:
        st.success(f"**Risk Level: {risk}**")

    if parsed.get('what_happened'):
        st.markdown("**📝 What Happened**")
        st.write(parsed['what_happened'])
    if parsed.get('impact'):
        st.markdown("**⚠️ Impact**")
        st.warning(parsed['impact'])
    if parsed.get('action'):
        st.markdown("**✅ Recommended Action**")
        st.info(parsed['action'])
    if parsed.get('context'):
        with st.expander("📚 Context"):
            st.write(parsed['context'])

# ============================================
# PDF HELPERS
# ============================================
def _clean_excerpt(text, max_len=260):
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_len:
        return text
    window = text[:max_len]
    last_period = window.rfind('. ')
    if last_period > max_len * 0.4:
        return window[:last_period + 1]
    last_space = window.rfind(' ')
    if last_space > 0:
        window = window[:last_space]
    return window.rstrip('.,;: ') + "…"

_RISK_HEX = {'Critical': '#E5484D', 'High': '#E5484D', 'Medium': '#D97706', 'Low': '#2F9E62'}

# ============================================
# GENERATE PDF REPORT
# ============================================
def generate_pdf(log_df, ai_summaries):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError as e:
        st.warning(f"PDF libraries not available: {e}")
        return b""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.6*inch, bottomMargin=0.6*inch,
                             leftMargin=0.7*inch, rightMargin=0.7*inch)
    styles = getSampleStyleSheet()
    story = []

    PRIMARY = colors.HexColor('#3B82C4')
    PRIMARY_DARK = colors.HexColor('#2C6B9E')
    TEXT = colors.HexColor('#16202B')
    MUTED = colors.HexColor('#7C8AA0')
    DANGER = colors.HexColor('#E5484D')
    SUCCESS = colors.HexColor('#2F9E62')
    LIGHT_BG = colors.HexColor('#F6F9FC')
    CALLOUT_BG = colors.HexColor('#FAFCFE')

    banner_title_style = ParagraphStyle('BannerTitle', fontName='Helvetica-Bold', fontSize=26,
                                         textColor=colors.white, alignment=TA_CENTER, leading=30)
    banner_subtitle_style = ParagraphStyle('BannerSubtitle', fontName='Helvetica', fontSize=13,
                                            textColor=colors.HexColor('#DCEBFA'), alignment=TA_CENTER, leading=16)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=15,
                                    spaceAfter=6, spaceBefore=4, textColor=TEXT, fontName='Helvetica-Bold')
    subheading_style = ParagraphStyle('SubHeading', fontSize=11, fontName='Helvetica-Bold',
                                       spaceAfter=3, textColor=PRIMARY_DARK)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, leading=14, textColor=TEXT)
    body_center_style = ParagraphStyle('BodyCenter', parent=body_style, alignment=TA_CENTER)
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=MUTED, alignment=TA_CENTER)

    banner = Table(
        [[Paragraph("INTRUSION DETECTION SYSTEM", banner_title_style)],
         [Paragraph("Security Analysis Report", banner_subtitle_style)]],
        colWidths=[6.6 * inch]
    )
    banner.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PRIMARY_DARK),
        ('TOPPADDING', (0, 0), (0, 0), 22),
        ('BOTTOMPADDING', (0, 0), (0, 0), 4),
        ('TOPPADDING', (0, 1), (0, 1), 2),
        ('BOTTOMPADDING', (0, 1), (0, 1), 22),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(banner)
    story.append(Spacer(1, 0.6 * inch))

    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_center_style))
    story.append(Paragraph(f"<b>Report ID:</b> IDS-{datetime.now().strftime('%Y%m%d-%H%M%S')}", body_center_style))
    story.append(Spacer(1, 0.4 * inch))

    if len(log_df) > 0:
        total = len(log_df)
        attacks = len(log_df[log_df['Prediction'] == 'Attack'])
        correct = len(log_df[log_df['Correct'] == 'Yes'])
        accuracy = (correct / total * 100) if total > 0 else 0

        stats_data = [
            ["TOTAL SAMPLES", "ATTACKS DETECTED", "ACCURACY"],
            [str(total), str(attacks), f"{accuracy:.1f}%"]
        ]
        stats_table = Table(stats_data, colWidths=[2.0 * inch, 2.0 * inch, 2.0 * inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 1), (-1, 1), LIGHT_BG),
            ('FONTSIZE', (0, 1), (-1, 1), 16),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 1), (-1, 1), TEXT),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#D6E4F0')),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor('#D6E4F0')),
        ]))
        story.append(stats_table)

    story.append(Spacer(1, 0.8 * inch))
    story.append(Paragraph("Confidential — For authorized use only", footer_style))
    story.append(PageBreak())

    story.append(Paragraph("Executive Summary", heading_style))

    if len(log_df) > 0:
        total = len(log_df)
        attacks = len(log_df[log_df['Prediction'] == 'Attack'])
        correct = len(log_df[log_df['Correct'] == 'Yes'])
        accuracy = (correct / total * 100) if total > 0 else 0
        false_positives = len(log_df[(log_df['Prediction'] == 'Attack') & (log_df['Correct'] == 'No')])
        false_negatives = len(log_df[(log_df['Prediction'] == 'Benign') & (log_df['Correct'] == 'No')])

        story.append(Paragraph(
            f"This report summarizes {total} analyzed network flows from the CICIDS2017 dataset. "
            f"The system identified {attacks} flows as malicious and {total - attacks} as benign, using a "
            f"Random Forest model with SHAP explainability for transparent, interpretable predictions.",
            body_style
        ))
        story.append(Spacer(1, 10))

        kpi_data = [
            ["Accuracy", "Correct", "False Positives", "False Negatives"],
            [f"{accuracy:.1f}%", f"{correct}/{total}", str(false_positives), str(false_negatives)]
        ]
        kpi_table = Table(kpi_data, colWidths=[1.55 * inch] * 4)
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EAF3FC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), PRIMARY_DARK),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8.5),
            ('FONTSIZE', (0, 1), (-1, 1), 13),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 1), (-1, 1), TEXT),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#D6E4F0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D6E4F0')),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ]))
        story.append(kpi_table)
    else:
        story.append(Paragraph("No data available for analysis.", body_style))

    story.append(Spacer(1, 14))

    if ai_summaries:
        parsed = parse_ai_summary(ai_summaries[0])
        if parsed.get('what_happened'):
            story.append(Paragraph("Key Finding", subheading_style))
            story.append(Paragraph(_clean_excerpt(parsed['what_happened'], 280), body_style))
            story.append(Spacer(1, 8))
        if parsed.get('action'):
            story.append(Paragraph("Recommended Action", subheading_style))
            story.append(Paragraph(_clean_excerpt(parsed['action'], 280), body_style))
            story.append(Spacer(1, 8))
        if parsed.get('risk'):
            risk_hex = _RISK_HEX.get(parsed['risk'], '#D97706')
            story.append(Paragraph(f"<font color='{risk_hex}'><b>Risk Level: {parsed['risk']}</b></font>", body_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Full details of the most recently detected attack, including the complete AI-generated analysis, follow on the next page.",
        ParagraphStyle('Note', parent=body_style, textColor=MUTED, fontSize=9)
    ))

    story.append(PageBreak())

    if ai_summaries:
        story.append(Paragraph("Detailed AI Triage Analysis", heading_style))
        story.append(Paragraph("Full AI-generated analysis for the most recent detected attack.", body_style))
        story.append(Spacer(1, 10))

        parsed = parse_ai_summary(ai_summaries[0])

        def callout(label, content, accent):
            if not content:
                return
            tbl = Table(
                [[Paragraph(f"<b>{label}</b><br/>{content}", body_style)]],
                colWidths=[6.6 * inch]
            )
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), CALLOUT_BG),
                ('LINEBEFORE', (0, 0), (0, 0), 3, accent),
                ('TOPPADDING', (0, 0), (0, 0), 10),
                ('BOTTOMPADDING', (0, 0), (0, 0), 10),
                ('LEFTPADDING', (0, 0), (0, 0), 14),
                ('RIGHTPADDING', (0, 0), (0, 0), 10),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 8))

        callout("What Happened", parsed.get('what_happened', ''), PRIMARY)
        callout("Impact", parsed.get('impact', ''), DANGER)
        callout("Recommended Action", parsed.get('action', ''), SUCCESS)
        callout("Context", parsed.get('context', ''), colors.HexColor('#D97706'))

        if parsed.get('risk'):
            story.append(Paragraph(f"<b>Risk Level:</b> {parsed['risk']}", body_style))
            story.append(Spacer(1, 4))

        if len(log_df) > 0:
            latest = log_df.iloc[-1]
            story.append(Paragraph(f"<b>Confidence:</b> {latest['Confidence']}  &nbsp;&nbsp;|&nbsp;&nbsp;  <b>MITRE Technique:</b> {latest['MITRE']}", body_style))

    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("Confidential — For authorized use only", footer_style))
    story.append(Paragraph(f"Report ID: IDS-{datetime.now().strftime('%Y%m%d-%H%M%S')}", footer_style))

    try:
        doc.build(story)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"PDF generation error: {e}")
        return b""

# ============================================
# WELCOME SCREEN
# ============================================
if not st.session_state.welcome_dismissed:

    st.markdown("""
    <div class="welcome-card">
        <h1>🛡️ Intrusion Detection System</h1>
        <p class="subtitle">AI-powered network attack detection with SHAP explainability</p>
        <div class="stat-row">
            <div class="stat"><div class="num">14</div><div class="lbl">Attack Types</div></div>
            <div class="stat"><div class="num">514K</div><div class="lbl">Test Flows</div></div>
            <div class="stat"><div class="num">100%</div><div class="lbl">Local</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-section">
        <h3>🤖 Model Selection</h3>
        <table class="model-table">
            <tr><th>Model</th><th>Type</th><th>Why Selected</th></tr>
            <tr class="selected">
                <td><b>Random Forest</b></td>
                <td>Ensemble</td>
                <td>Best balance of accuracy and interpretability; handles tabular data well</td>
            </tr>
            <tr>
                <td>Isolation Forest</td>
                <td>Unsupervised</td>
                <td>Anomaly detection baseline (lower performance)</td>
            </tr>
            <tr>
                <td>Autoencoder</td>
                <td>Neural Network</td>
                <td>Unsupervised anomaly detection (complementary)</td>
            </tr>
        </table>
        <p class="model-note">✅ <b>Selected:</b> Random Forest — trained on 2.57M flows, validated on held-out test set</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-section">
        <h3>📋 What This System Does</h3>
        <p>This system detects network attacks using a Random Forest model trained on the CICIDS2017 dataset.</p>
        <ul>
            <li><b>Input:</b> Network flow features (60 dimensions)</li>
            <li><b>Output:</b> Attack or Benign classification</li>
            <li><b>Explainability:</b> SHAP values show which features influenced each prediction</li>
            <li><b>Triage:</b> AI-generated summaries for actionable insights</li>
        </ul>
        <p>The model is evaluated on a held-out test set (20% of data) to ensure honest performance reporting.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="welcome-btn-row">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Get Started", type="primary", use_container_width=True):
            st.session_state.welcome_dismissed = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ============================================
# MAIN DASHBOARD
# ============================================
st.markdown('<h1 style="text-align:center;">🛡️ Intrusion Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#45566B;">Machine-learning network attack detection with SHAP explainability</p>', unsafe_allow_html=True)
st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)

st.sidebar.header("📊 Dashboard")
st.sidebar.write(f"Alerts: {len(st.session_state.alerts)}")

if st.session_state.history:
    total = len(st.session_state.history)
    correct = sum(1 for h in st.session_state.history if h.get('is_correct', False))
    st.sidebar.metric("Analyzed", total)
    st.sidebar.metric("Correct", f"{correct}/{total}")

st.sidebar.markdown("---")

if st.sidebar.button("📄 Generate PDF Report", type="primary", use_container_width=True):
    if len(st.session_state.log_df) > 0:
        with st.spinner("Generating professional PDF report..."):
            pdf = generate_pdf(st.session_state.log_df, st.session_state.ai_summaries)
            if pdf and len(pdf) > 100:
                st.sidebar.success("✅ PDF generated!")
                st.sidebar.download_button(
                    label="⬇️ Download Report",
                    data=pdf,
                    file_name=f"ids_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.sidebar.error("PDF generation failed. Please check logs.")
    else:
        st.sidebar.warning("No data to report")

if len(st.session_state.log_df) > 0:
    csv = st.session_state.log_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="⬇️ Download Logs (CSV)",
        data=csv,
        file_name=f"ids_logs_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.sidebar.markdown("---")
st.sidebar.caption("🔒 All processing is local")
st.sidebar.caption(f"📁 Logs: {LOG_FILE}")

# ============================================
# TABS
# ============================================
tab1, tab2 = st.tabs(["🔎 Threat Analysis", "📋 Logs Panel"])

with tab1:
    if not st.session_state.models_loaded:
        st.warning("Models not loaded.")
        st.stop()

    st.markdown('<h3 class="section-title">Analyze Test Set Sample</h3>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center; color:#7C8AA0;">Test set: {len(test_indices):,} held-out flows</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("🎲 Random Attack", use_container_width=True):
            attack_idx = [i for i in test_indices if y_full[i] == 1]
            if attack_idx:
                idx = np.random.choice(attack_idx)
                X_row = X_full.iloc[[idx]].values
                attack_type = attack_labels[idx]
                result = analyze_sample(X_row, f"Test #{idx}", attack_type)
                st.session_state.latest = result
                st.rerun()
            else:
                st.warning("No attacks in test set")

    with c2:
        if st.button("✅ Random Benign", use_container_width=True):
            benign_idx = [i for i in test_indices if y_full[i] == 0]
            if benign_idx:
                idx = np.random.choice(benign_idx)
                X_row = X_full.iloc[[idx]].values
                result = analyze_sample(X_row, f"Test #{idx} (benign)", "BENIGN")
                st.session_state.latest = result
                st.rerun()
            else:
                st.warning("No benign in test set")

    with c3:
        options = st.session_state.sample_test_options or []
        labels = ["— select a sample —"] + [label for _, label in options]

        selected = st.selectbox(
            "Pick a test row (or enter row number below):",
            labels,
            help="Select a pre-populated sample, or enter any row number in the text box below"
        )

        manual_row = st.number_input(
            "Or enter custom row number:",
            min_value=0,
            max_value=len(X_full) - 1,
            value=None,
            step=1,
            format="%d",
            help="Enter any row index from the test set to analyze"
        )

        row_to_analyze = None
        source_label = None

        if selected != "— select a sample —":
            idx = next((i for i, label in options if label == selected), None)
            if idx is not None:
                row_to_analyze = idx
                source_label = f"Test #{idx}"

        if manual_row is not None and manual_row >= 0 and manual_row < len(X_full):
            if manual_row in test_indices:
                row_to_analyze = manual_row
                source_label = f"Test #{manual_row}"
            else:
                st.warning(f"Row {manual_row} is not in the test set. Please enter a row from the test set.")

        if row_to_analyze is not None:
            if st.button("🔍 Analyze Selected", use_container_width=True):
                X_row = X_full.iloc[[row_to_analyze]].values
                attack_type = attack_labels[row_to_analyze]
                result = analyze_sample(X_row, source_label, attack_type)
                st.session_state.latest = result
                st.rerun()

    st.markdown("---")

    if st.session_state.latest:
        render_result(st.session_state.latest)
    else:
        st.info("👆 Click a button above or select a row to analyze a sample.")

with tab2:
    st.markdown('<h3 class="section-title">📋 Logs & History</h3>', unsafe_allow_html=True)

    log_df = st.session_state.log_df

    if len(log_df) > 0:
        total = len(log_df)
        attacks = len(log_df[log_df['Prediction'] == 'Attack'])
        correct = len(log_df[log_df['Correct'] == 'Yes'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples", total)
        c2.metric("Attacks", attacks)
        c3.metric("Correct", correct)
        c4.metric("Accuracy", f"{correct/total*100:.1f}%" if total > 0 else "0%")

        st.markdown("---")

        st.markdown('<h4 class="section-title">Filter Logs</h4>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1:
            pred_filter = st.selectbox("Prediction", ["All", "Attack", "Benign"])
        with f2:
            truth_filter = st.selectbox("Ground Truth", ["All"] + sorted(log_df['Ground_Truth'].unique().tolist()))
        with f3:
            correct_filter = st.selectbox("Correct", ["All", "Yes", "No"])

        filtered = log_df.copy()
        if pred_filter != "All":
            filtered = filtered[filtered['Prediction'] == pred_filter]
        if truth_filter != "All":
            filtered = filtered[filtered['Ground_Truth'] == truth_filter]
        if correct_filter != "All":
            filtered = filtered[filtered['Correct'] == correct_filter]

        st.dataframe(filtered.iloc[::-1], use_container_width=True, height=400)
        st.caption(f"Showing {len(filtered)} of {total} entries")

        st.markdown("---")
        st.markdown('<h4 class="section-title">Alert Timeline</h4>', unsafe_allow_html=True)
        if st.session_state.alerts:
            for alert in reversed(st.session_state.alerts[-10:]):
                with st.expander(f"🚨 {alert['timestamp']} — {alert.get('attack_type', 'Unknown')} ({alert['confidence']*100:.0f}%)"):
                    if 'ai_summary' in alert:
                        render_ai_summary(alert['ai_summary'])
                    else:
                        st.caption("No AI summary")
        else:
            st.caption("No alerts recorded")

        st.markdown("---")
        if st.button("🗑️ Clear All Logs", type="secondary"):
            st.session_state.history = []
            st.session_state.alerts = []
            st.session_state.ai_summaries = []
            st.session_state.latest = None
            st.session_state.show_shap = False
            st.session_state.show_ai_summary = False
            st.session_state.cached_shap_fig = None
            st.session_state.cached_ai_summary = None
            st.session_state.log_df = pd.DataFrame(columns=[
                'Timestamp', 'Source', 'Prediction', 'Confidence',
                'Ground_Truth', 'MITRE', 'Correct', 'Top_Features'
            ])
            if os.path.exists(LOG_FILE):
                try:
                    os.remove(LOG_FILE)
                except:
                    pass
            st.rerun()
    else:
        st.info("No logs yet. Analyze some samples to populate this panel.")

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    🔒 All processing is local · CICIDS2017 · 14 attack types · 2.57M training flows
</div>
""", unsafe_allow_html=True)
