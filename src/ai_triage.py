import pandas as pd
import numpy as np
import pickle
import json
import requests
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("AI TRIAGE WITH OLLAMA")
print("=" * 60)

# ============================================
# 1. LOAD MODELS AND DATA
# ============================================
print("\n📊 Loading models and data...")

with open('models/random_forest_multiday.pkl', 'rb') as f:
    rf_model = pickle.load(f)

import shap
explainer = shap.TreeExplainer(rf_model)

X_full = pd.read_csv('data/processed/X_processed_multiday.csv')
feature_names = X_full.columns.tolist()

test_indices = pd.read_csv('data/processed/test_indices.csv').values.ravel()

X_test = X_full.iloc[test_indices]
y_test = pd.read_csv('data/processed/y_processed_multiday.csv').values.ravel()[test_indices]

attack_df = pd.read_csv('data/processed/attack_types_multiday.csv')
attack_labels = attack_df['attack_type'].values[test_indices]

print(f"   ✅ Random Forest loaded")
print(f"   ✅ SHAP explainer loaded")
print(f"   ✅ {len(feature_names)} features")
print(f"   ✅ Test set loaded: {len(X_test):,} rows")

# ============================================
# 2. IMPORT MITRE MAPPING
# ============================================
print("\n📊 Loading MITRE mapping...")

try:
    from mitre_mapping import map_to_mitre, techniques, classify_attack_technique
    print(f"   ✅ MITRE mapping loaded")
except ImportError as e:
    print(f"   ❌ Failed to import MITRE mapping: {e}")
    print("   Creating fallback...")
    techniques = {'T1499': {'name': 'Unknown Attack'}}
    
    def classify_attack_technique(top_features):
        return 'T1499'
    
    def map_to_mitre(behavior):
        return 'T1499'

# ============================================
# 3. LOAD SAMPLE ATTACK
# ============================================
print("\n📊 Loading sample attack...")

attack_indices = np.where(y_test == 1)[0]

if len(attack_indices) == 0:
    print("   ❌ No attacks found in test set")
    sys.exit(1)

sample_idx = attack_indices[0]
X_sample = X_test.iloc[sample_idx].values.reshape(1, -1)
attack_type = attack_labels[sample_idx]

print(f"   ✅ Sample attack found (index: {sample_idx})")
print(f"   ✅ Attack type: {attack_type}")

# ============================================
# 4. PREDICTION AND SHAP VALUES
# ============================================
print("\n🔍 Getting prediction and SHAP explanation...")

prediction = rf_model.predict(X_sample)[0]
prediction_proba = rf_model.predict_proba(X_sample)[0]

print(f"   Prediction: {'ATTACK' if prediction == 1 else 'BENIGN'}")
print(f"   Confidence: {prediction_proba[prediction]:.2%}")

shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

shap_importance = np.abs(shap_values).flatten()
top_indices = np.argsort(shap_importance)[-5:][::-1]

top_features = []
for idx in top_indices:
    if shap_importance[idx] > 0.01:
        feature_name = feature_names[idx]
        feature_value = float(X_sample[0][idx])
        shap_value = float(shap_values[0][idx])
        top_features.append({
            'feature': feature_name,
            'value': feature_value,
            'shap': shap_value,
            'direction': 'attack' if shap_value > 0 else 'benign'
        })

print("   Top contributing features:")
for f in top_features:
    direction = f['direction']
    print(f"   • {f['feature']}: {f['value']:.2f} (SHAP: {f['shap']:.3f} → {direction})")

# ============================================
# 5. MITRE MAPPING
# ============================================
print("\n📊 Mapping to MITRE ATT&CK...")

try:
    mitre_technique = classify_attack_technique(top_features)
    tech_info = techniques.get(mitre_technique, {'name': 'Unknown Attack'})
except NameError:
    mitre_technique = 'T1499'
    tech_info = {'name': 'Unknown Attack'}

mitre_display = f"{mitre_technique} ({tech_info['name']})"
print(f"   MITRE Technique: {mitre_display}")

# ============================================
# 6. BUILD OLLAMA PROMPT
# ============================================
print("\n🤖 Building AI triage prompt...")

feature_list = '\n'.join([f"- {f['feature']}: {f['value']:.2f} (SHAP: {f['shap']:.3f}, pushes toward {f['direction']})" for f in top_features])

prompt = f"""You are a senior SOC analyst explaining a security detection to a junior analyst.

ALERT DETAILS:
- Classification: {'ATTACK' if prediction == 1 else 'BENIGN'}
- Confidence: {prediction_proba[prediction]:.2%}
- MITRE Technique: {mitre_display}
- Attack Type: {attack_type}

TOP CONTRIBUTING FEATURES:
{feature_list}

REQUIRED OUTPUT — Write in clear, plain English:

1. **WHAT HAPPENED?** (1-2 sentences explaining the detection)
2. **WHY SHOULD I CARE?** (1 sentence on impact)
3. **WHAT SHOULD I DO NOW?** (2-3 actionable steps)
4. **THE BIGGER PICTURE** (1 sentence on context)

Write naturally, like talking to a colleague. Keep it under 150 words.
"""

print("   ✅ Prompt built")

# ============================================
# 7. CALL OLLAMA
# ============================================
print("\n🔄 Calling Ollama (llama3.2)...")

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
        result = response.json()
        triage_summary = result.get('response', '')
        print("\n" + "=" * 60)
        print("AI TRIAGE SUMMARY")
        print("=" * 60)
        print(triage_summary)
        print("=" * 60)

        os.makedirs('models', exist_ok=True)
        with open('models/ai_triage_sample.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("AI TRIAGE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"ALERT: {'ATTACK' if prediction == 1 else 'BENIGN'}\n")
            f.write(f"ATTACK TYPE: {attack_type}\n")
            f.write(f"MITRE: {mitre_display}\n")
            f.write(f"CONFIDENCE: {prediction_proba[prediction]:.2%}\n\n")
            f.write(triage_summary)

        print("\n   💾 Saved to models/ai_triage_sample.txt")

    else:
        print(f"   ❌ Ollama error: {response.status_code}")

except requests.exceptions.ConnectionError:
    print("   ❌ Could not connect to Ollama.")
    print("   Make sure Ollama is running: 'ollama serve'")

except Exception as e:
    print(f"   ❌ Error: {e}")

# ============================================
# 8. SUMMARY
# ============================================
print("\n" + "=" * 60)
print("AI TRIAGE SUMMARY")
print("=" * 60)
print(f"   Sample analyzed: Attack #{sample_idx}")
print(f"   Attack Type: {attack_type}")
print(f"   Prediction: {'ATTACK' if prediction == 1 else 'BENIGN'}")
print(f"   MITRE: {mitre_display}")
print(f"   Triage saved: models/ai_triage_sample.txt")

print("\n" + "=" * 60)
print("✅ AI triage complete!")
print("=" * 60)