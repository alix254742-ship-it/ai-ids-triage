import pandas as pd
import numpy as np
import os
import glob

print("=" * 60)
print("DATA PREPROCESSING PIPELINE")
print("=" * 60)

# ============================================
# CONFIGURATION
# ============================================
RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"

# ============================================
# STEP 1: LOAD ALL CSV FILES
# ============================================
print(f"\n📊 Loading CSV files from {RAW_DIR}...")

csv_files = glob.glob(RAW_DIR + "*.csv")
print(f"   Found {len(csv_files)} CSV files")

dfs = []
total_rows = 0

for filepath in csv_files:
    filename = os.path.basename(filepath)
    print(f"\n   📄 Loading: {filename}")
    
    df = pd.read_csv(filepath)
    
    # Clean column names
    df.columns = df.columns.str.replace(' ', '')
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN
    rows_before = len(df)
    df = df.dropna()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    print(f"      Rows: {len(df):,} (removed {rows_before - len(df):,})")
    dfs.append(df)
    total_rows += len(df)

if not dfs:
    print("\n❌ No data files found!")
    exit(1)

# ============================================
# STEP 2: COMBINE ALL DATAFRAMES
# ============================================
print(f"\n📊 Combining {len(dfs)} files...")
df = pd.concat(dfs, ignore_index=True)
print(f"   Total rows: {len(df):,}")

# ============================================
# STEP 3: ENCODE LABELS
# ============================================
print("\n🔤 Encoding labels...")
df['Label_encoded'] = df['Label'].apply(lambda x: 1 if x != 'BENIGN' else 0)

attack_count = df['Label_encoded'].sum()
benign_count = len(df) - attack_count

print(f"   Attacks: {attack_count:,} ({attack_count/len(df)*100:.1f}%)")
print(f"   Benign:  {benign_count:,} ({benign_count/len(df)*100:.1f}%)")

print("\n   Attack types found:")
attack_types = df[df['Label_encoded'] == 1]['Label'].value_counts()
for label, count in attack_types.items():
    print(f"      {label}: {count:,}")

# ============================================
# STEP 4: DROP LOW-VARIANCE COLUMNS
# ============================================
print("\n🧹 Dropping low-variance columns...")

cols_to_drop = []
for col in df.columns:
    if col in ['Label', 'Label_encoded']:
        continue
    if df[col].nunique() <= 2:
        cols_to_drop.append(col)

if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"   Dropped {len(cols_to_drop)} low-variance columns")
    print(f"   Remaining columns: {len(df.columns)}")

# ============================================
# STEP 5: SAVE PROCESSED DATA
# ============================================
print("\n💾 Saving processed data...")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Save attack type labels
attack_type_df = pd.DataFrame({
    'attack_type': df['Label'].values,
    'is_attack': df['Label_encoded'].values
})
attack_type_df.to_csv(PROCESSED_DIR + 'attack_types_multiday.csv', index=False)
print(f"   ✅ Saved attack types: attack_types_multiday.csv ({len(attack_type_df):,} rows)")

# Save features and labels
X = df.drop(['Label', 'Label_encoded'], axis=1)
y = df['Label_encoded']

X.to_csv(PROCESSED_DIR + 'X_processed_multiday.csv', index=False)
y.to_csv(PROCESSED_DIR + 'y_processed_multiday.csv', index=False)

print(f"   ✅ Saved features: X_processed_multiday.csv ({X.shape[1]} features, {X.shape[0]:,} rows)")
print(f"   ✅ Saved labels: y_processed_multiday.csv ({y.shape[0]:,} rows)")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)
print(f"   Total rows: {len(df):,}")
print(f"   Total columns: {len(df.columns)}")
print(f"   Attack ratio: {attack_count/len(df):.3f}")
print(f"\n📁 Files saved:")
print(f"   - X_processed_multiday.csv (features)")
print(f"   - y_processed_multiday.csv (binary labels)")
print(f"   - attack_types_multiday.csv (attack type labels)")

print("\n" + "=" * 60)
print("✅ Preprocessing complete!")
print("=" * 60)