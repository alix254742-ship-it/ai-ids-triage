import pandas as pd
import numpy as np
import time
import pickle
import os
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("MODEL TRAINING PIPELINE")
print("=" * 60)

# ============================================
# GPU VERIFICATION
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🔍 GPU Status:")
print(f"   Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  GPU not available — running on CPU")

# ============================================
# 1. LOAD PROCESSED DATA
# ============================================
print("\n📊 Loading processed data...")
X = pd.read_csv('data/processed/X_processed_multiday.csv')
y = pd.read_csv('data/processed/y_processed_multiday.csv').values.ravel()
attack_df = pd.read_csv('data/processed/attack_types_multiday.csv')
attack_labels = attack_df['attack_type'].values

print(f"   Full dataset: {X.shape}")
print(f"   Overall attack ratio: {y.mean():.3f}")

print("\n📊 Class distribution:")
print(f"   BENIGN: {len(y) - y.sum():,} ({((len(y) - y.sum())/len(y))*100:.1f}%)")
print(f"   ATTACKS: {y.sum():,} ({y.mean()*100:.1f}%)")

print("\n   Attack type distribution:")
attack_counts = pd.Series(attack_labels[y == 1]).value_counts()
for atype, count in attack_counts.items():
    print(f"      {atype}: {count:,} ({count/y.sum()*100:.1f}%)")

# ============================================
# 2. TRAIN/TEST SPLIT
# ============================================
print("\n📊 Splitting data (80/20 stratified)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_idx = X_train.index
X_test_idx = X_test.index
attack_labels_train = attack_labels[X_train_idx]
attack_labels_test = attack_labels[X_test_idx]

print(f"   Train set: {len(X_train):,} rows")
print(f"   Test set: {len(X_test):,} rows")

# ============================================
# 3. CLASS BALANCING WITH SMOTE
# ============================================
print("\n" + "=" * 60)
print("CLASS BALANCING (SMOTE)")
print("=" * 60)

print(f"\n   Before SMOTE: {len(X_train):,} rows, {y_train.sum():,} attacks")

smote = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"   After SMOTE: {len(X_train_resampled):,} rows, {y_train_resampled.sum():,} attacks")

# ============================================
# 4. CONVERT TO NUMPY
# ============================================
X_train_np = X_train_resampled.values.astype(np.float32)
y_train_np = y_train_resampled.astype(np.int32)
X_test_np = X_test.values.astype(np.float32)
y_test_np = y_test.astype(np.int32)

print(f"\n   Train attack ratio: {y_train_np.mean():.3f}")
print(f"   Test attack ratio:  {y_test_np.mean():.3f}")

results = {}

# ============================================
# 5. MODEL 1: ISOLATION FOREST
# ============================================
print("\n" + "=" * 60)
print("Training Isolation Forest")
print("=" * 60)

start_time = time.time()

iso_forest = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=y_train_np.mean(),
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_train_np)
iso_pred = iso_forest.predict(X_test_np)
iso_pred_binary = np.where(iso_pred == -1, 1, 0)

iso_time = time.time() - start_time

iso_accuracy = accuracy_score(y_test_np, iso_pred_binary)
iso_precision = precision_score(y_test_np, iso_pred_binary)
iso_recall = recall_score(y_test_np, iso_pred_binary)
iso_f1 = f1_score(y_test_np, iso_pred_binary)

results['Isolation Forest'] = {
    'accuracy': iso_accuracy, 'precision': iso_precision,
    'recall': iso_recall, 'f1': iso_f1, 'time': iso_time
}

print(f"✅ Trained in {iso_time:.2f} seconds")
print(f"   Accuracy: {iso_accuracy:.4f}  Precision: {iso_precision:.4f}  Recall: {iso_recall:.4f}  F1: {iso_f1:.4f}")

os.makedirs('models', exist_ok=True)
with open('models/isolation_forest_multiday.pkl', 'wb') as f:
    pickle.dump(iso_forest, f)
print("   💾 Saved: isolation_forest_multiday.pkl")

# ============================================
# 6. MODEL 2: RANDOM FOREST
# ============================================
print("\n" + "=" * 60)
print("Training Random Forest")
print("=" * 60)

start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model.fit(X_train_np, y_train_np)
rf_pred = rf_model.predict(X_test_np)

rf_time = time.time() - start_time

rf_accuracy = accuracy_score(y_test_np, rf_pred)
rf_precision = precision_score(y_test_np, rf_pred)
rf_recall = recall_score(y_test_np, rf_pred)
rf_f1 = f1_score(y_test_np, rf_pred)

results['Random Forest'] = {
    'accuracy': rf_accuracy, 'precision': rf_precision,
    'recall': rf_recall, 'f1': rf_f1, 'time': rf_time
}

print(f"✅ Trained in {rf_time:.2f} seconds")
print(f"   Accuracy: {rf_accuracy:.4f}  Precision: {rf_precision:.4f}  Recall: {rf_recall:.4f}  F1: {rf_f1:.4f}")

print("\n   🔁 Running 5-fold cross-validation...")
cv_start = time.time()
cv_scores = cross_val_score(rf_model, X_train_np, y_train_np, cv=5, scoring='f1', n_jobs=-1)
cv_time = time.time() - cv_start
print(f"   5-fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  (took {cv_time:.1f}s)")

results['Random Forest']['cv_f1_mean'] = cv_scores.mean()
results['Random Forest']['cv_f1_std'] = cv_scores.std()

with open('models/random_forest_multiday.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("   💾 Saved: random_forest_multiday.pkl")

# ============================================
# 7. MODEL 3: AUTOENCODER
# ============================================
print("\n" + "=" * 60)
print("Training Autoencoder (benign-only)")
print("=" * 60)

X_train_benign = X_train[y_train == 0].values.astype(np.float32)
print(f"   Training on benign-only data: {len(X_train_benign):,} rows")

scaler = StandardScaler()
X_train_benign_scaled = scaler.fit_transform(X_train_benign)
print(f"   Data scaled (mean≈0, std≈1)")

X_tensor = torch.FloatTensor(X_train_benign_scaled).to(device)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=20):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

input_dim = X_train_np.shape[1]
autoencoder = Autoencoder(input_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

epochs = 30
batch_size = 128

dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

start_time = time.time()
print(f"   Training: {epochs} epochs, batch_size={batch_size}")

for epoch in range(epochs):
    epoch_loss = 0
    for batch in dataloader:
        batch_data = batch[0].to(device)
        optimizer.zero_grad()
        reconstructed = autoencoder(batch_data)
        loss = criterion(reconstructed, batch_data)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        avg_loss = epoch_loss / len(dataloader)
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

autoencoder_time = time.time() - start_time

autoencoder.eval()
with torch.no_grad():
    train_benign_recon = autoencoder(X_tensor)
    train_benign_error = torch.mean((train_benign_recon - X_tensor) ** 2, dim=1)

threshold = np.percentile(train_benign_error.cpu().numpy(), 95)
print(f"   Anomaly threshold (95th percentile): {threshold:.6f}")

X_test_scaled = scaler.transform(X_test_np)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

with torch.no_grad():
    test_recon = autoencoder(X_test_tensor)
    test_recon_error = torch.mean((test_recon - X_test_tensor) ** 2, dim=1)

ae_pred = (test_recon_error.cpu().numpy() > threshold).astype(int)

ae_accuracy = accuracy_score(y_test_np, ae_pred)
ae_precision = precision_score(y_test_np, ae_pred)
ae_recall = recall_score(y_test_np, ae_pred)
ae_f1 = f1_score(y_test_np, ae_pred)

results['Autoencoder'] = {
    'accuracy': ae_accuracy, 'precision': ae_precision,
    'recall': ae_recall, 'f1': ae_f1, 'time': autoencoder_time
}

print(f"\n✅ Trained in {autoencoder_time:.2f} seconds")
print(f"   Accuracy: {ae_accuracy:.4f}  Precision: {ae_precision:.4f}  Recall: {ae_recall:.4f}  F1: {ae_f1:.4f}")

torch.save(autoencoder.state_dict(), 'models/autoencoder_multiday.pth')
print("   💾 Saved: autoencoder_multiday.pth")

import joblib
joblib.dump(scaler, 'models/scaler_multiday.pkl')
print("   💾 Saved: scaler_multiday.pkl")

# ============================================
# 8. SAVE TEST INDICES
# ============================================
print("\n💾 Saving test indices...")

test_indices = X_test.index.tolist()
pd.Series(test_indices).to_csv('data/processed/test_indices.csv', index=False)
print(f"   ✅ Saved: test_indices.csv ({len(test_indices):,} rows)")

attack_labels_test_df = pd.DataFrame({
    'index': X_test_idx,
    'attack_type': attack_labels_test
})
attack_labels_test_df.to_csv('data/processed/attack_labels_test.csv', index=False)
print(f"   ✅ Saved: attack_labels_test.csv")

# ============================================
# 9. RESULTS SUMMARY
# ============================================
print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

print("\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
    "Model", "Accuracy", "Precision", "Recall", "F1 Score", "Time (s)"
))
print("-" * 75)

for model_name, metrics in results.items():
    print("{:<20} {:>9.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>11.2f}".format(
        model_name, metrics['accuracy'], metrics['precision'],
        metrics['recall'], metrics['f1'], metrics['time']
    ))

best_model = max(results, key=lambda x: results[x]['f1'])
print(f"\n🏆 Best model: {best_model} (F1 = {results[best_model]['f1']:.4f})")
if 'cv_f1_mean' in results['Random Forest']:
    print(f"   Random Forest 5-fold CV F1: {results['Random Forest']['cv_f1_mean']:.4f} ± {results['Random Forest']['cv_f1_std']:.4f}")

print("\n💾 Saving results...")
results_df = pd.DataFrame(results).T
results_df.to_csv('models/model_results_multiday.csv')
print("   ✅ Saved: model_results_multiday.csv")

print("\n" + "=" * 60)
print("✅ Model training complete!")
print("=" * 60)