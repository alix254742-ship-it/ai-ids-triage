import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os
import warnings
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

print("=" * 60)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 60)

# ============================================
# 1. LOAD DATA AND MODEL
# ============================================
print("\n📊 Loading data and model...")

with open('models/random_forest_multiday.pkl', 'rb') as f:
    rf_model = pickle.load(f)

print("   ✅ Random Forest model loaded")

X_full = pd.read_csv('data/processed/X_processed_multiday.csv')
y_full = pd.read_csv('data/processed/y_processed_multiday.csv').values.ravel()
test_indices = pd.read_csv('data/processed/test_indices.csv').values.ravel()

if len(test_indices) == 0:
    print("   ❌ No test indices found! Run train_models.py first.")
    exit(1)

X_test = X_full.iloc[test_indices]
y_test = y_full[test_indices]

print(f"   ✅ Test set loaded: {len(X_test):,} rows (20% hold-out)")

sample_size = 10000
np.random.seed(42)
if len(X_test) > sample_size:
    indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test.iloc[indices].values
    y_sample = y_test[indices]
else:
    X_sample = X_test.values
    y_sample = y_test

print(f"   Using {len(X_sample):,} samples for SHAP analysis")
print(f"   Sample attack ratio: {y_sample.mean():.3f}")

# ============================================
# 2. CREATE SHAP EXPLAINER
# ============================================
print("\n🔍 Creating SHAP explainer...")
explainer = shap.TreeExplainer(rf_model)
print("   ✅ SHAP TreeExplainer created")

# ============================================
# 3. CALCULATE SHAP VALUES
# ============================================
print("\n📊 Calculating SHAP values...")
print(f"   📦 Processing {len(X_sample):,} samples in batches of 500")
print(f"   ⏱️  Estimated time: 15-25 minutes on CPU")
print("   " + "-" * 50)

batch_size = 500
n_samples = len(X_sample)
n_batches = (n_samples + batch_size - 1) // batch_size

all_shap_values = []

pbar = tqdm(total=n_samples, desc="SHAP Progress", unit="samples",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

start_time = time.time()

for i in range(0, n_samples, batch_size):
    batch_end = min(i + batch_size, n_samples)
    batch = X_sample[i:batch_end]
    
    batch_shap = explainer.shap_values(batch)
    
    if isinstance(batch_shap, list):
        batch_shap = batch_shap[1]
    elif len(batch_shap.shape) == 3:
        batch_shap = batch_shap[:, :, 1]
    
    all_shap_values.append(batch_shap)
    pbar.update(batch_end - i)
    
    if (i // batch_size) % 2 == 0:
        pbar.set_postfix({"Batch": f"{i//batch_size + 1}/{n_batches}"})

pbar.close()

shap_values = np.vstack(all_shap_values)

end_time = time.time()
elapsed = end_time - start_time

print(f"\n   ⏱️  SHAP calculation completed in {elapsed/60:.1f} minutes")
print(f"   ✅ SHAP values calculated: {shap_values.shape[0]} samples")

# ============================================
# 4. GLOBAL FEATURE IMPORTANCE
# ============================================
print("\n📊 Global feature importance (top 10):")

feature_importance = np.abs(shap_values).mean(axis=0)
feature_names = X_test.columns

sorted_idx = np.argsort(feature_importance)[::-1]

print("\n   Top 10 features by mean |SHAP|:")
print("   " + "-" * 50)
for i in range(10):
    idx = sorted_idx[i]
    print(f"   {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")

# ============================================
# 5. SAVE SHAP PLOTS
# ============================================
print("\n💾 Saving SHAP plots...")

os.makedirs('screenshots', exist_ok=True)

print("   📊 Generating summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('screenshots/shap_summary_plot_multiday.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: shap_summary_plot_multiday.png")

attack_indices = np.where(y_sample == 1)[0]
if len(attack_indices) > 0:
    idx = attack_indices[0]
    print("   📊 Generating waterfall plot...")
    plt.figure(figsize=(10, 6))
    
    if isinstance(explainer.expected_value, (list, np.ndarray)) and np.ndim(explainer.expected_value) > 0:
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value
    
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=base_value,
            data=X_sample[idx],
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig('screenshots/shap_waterfall_attack_multiday.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: shap_waterfall_attack_multiday.png")

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_SHAP': feature_importance
}).sort_values('Mean_SHAP', ascending=False)

importance_df.to_csv('models/shap_feature_importance_multiday.csv', index=False)
print("   ✅ Saved: shap_feature_importance_multiday.csv")

# ============================================
# 6. SUMMARY
# ============================================
print("\n" + "=" * 60)
print("SHAP ANALYSIS SUMMARY")
print("=" * 60)

print(f"   Total samples analyzed: {len(X_sample):,}")
print(f"   Total features: {len(feature_names)}")
print(f"   Top feature: {feature_names[sorted_idx[0]]}")
print(f"   Sample attack ratio: {y_sample.mean():.3f}")
print(f"   Time taken: {elapsed/60:.1f} minutes")
print(f"   Saved plots: shap_summary_plot_multiday.png, shap_waterfall_attack_multiday.png")
print(f"   Saved data: shap_feature_importance_multiday.csv")

print("\n" + "=" * 60)
print("✅ SHAP analysis complete!")
print("=" * 60)