"""
=============================================================
UPDATED CELLS FOR AI_3D_Stress_Validator.ipynb
=============================================================
Copy-paste these into the corresponding cells in Google Colab.
=============================================================
"""

# ============================================================
# CELL 1.3  —  Discover STL files and load REAL stress data
# ============================================================
# Replace the ENTIRE existing cell 1.3 with this code:

# 1.3  Discover STL files and load REAL stress data from bracket_labels.csv
import glob
import csv
import numpy as np
import pandas as pd

# --- Find STL files ---
stl_files = sorted(glob.glob(os.path.join(DATA_ROOT, '**', '*.stl'), recursive=True))
print(f'Found {len(stl_files)} STL files total.')
stl_files = stl_files[:NUM_SAMPLES]
print(f'Using {len(stl_files)} samples.')

# --- Load real FEA stress data from bracket_labels.csv ---
# Search for the CSV in DATA_ROOT and Google Drive
csv_paths = (
    glob.glob(os.path.join(DATA_ROOT, '**', 'bracket_labels.csv'), recursive=True) +
    glob.glob('/content/drive/MyDrive/**/bracket_labels.csv', recursive=True)
)

if not csv_paths:
    raise FileNotFoundError(
        'bracket_labels.csv not found! Please upload it to '
        f'{DATA_ROOT}/ or your Google Drive.'
    )

csv_path = csv_paths[0]
print(f'\nLoading FEA data from: {csv_path}')
df = pd.read_csv(csv_path)
print(f'Loaded {len(df)} rows, {len(df.columns)} columns.')
print(f'Columns: {list(df.columns)}')

# --- Stress columns: max stress across 4 FEA load cases ---
STRESS_COLS = [
    'max_ver_stress(MPa)',   # Vertical load
    'max_hor_stress(MPa)',   # Horizontal load
    'max_dia_stress(MPa)',   # Diagonal load
    'max_tor_stress(MPa)',   # Torsional load
]

# Check that stress columns exist
missing = [c for c in STRESS_COLS if c not in df.columns]
if missing:
    print(f'\n⚠️  Missing columns: {missing}')
    print(f'Available columns: {list(df.columns)}')
    # Fallback: try to find any columns containing "stress"
    STRESS_COLS = [c for c in df.columns if 'stress' in c.lower()]
    print(f'Using fallback stress columns: {STRESS_COLS}')

# Compute the worst-case (max) stress across all load cases per part
df['max_stress_all'] = df[STRESS_COLS].max(axis=1)

# --- Labeling strategy ---
# Use the 75th percentile as the fail threshold.
# Parts above this are labeled FAIL (1), below are PASS (0).
# STRESS_FAIL_THRESHOLD (0.8) is used as the quantile here.
threshold = df['max_stress_all'].quantile(STRESS_FAIL_THRESHOLD)
df['label'] = (df['max_stress_all'] >= threshold).astype(float)

# Build stress lookup by item_name
stress_data = {}
label_data  = {}
for _, row in df.iterrows():
    name = str(row['item_name'])
    stress_data[name] = row['max_stress_all']
    label_data[name]  = row['label']

# --- Summary statistics ---
n_fail = int(df['label'].sum())
n_pass = len(df) - n_fail
print(f'\n📊 Stress Statistics:')
print(f'   Min stress  : {df["max_stress_all"].min():.2f} MPa')
print(f'   Mean stress : {df["max_stress_all"].mean():.2f} MPa')
print(f'   Median      : {df["max_stress_all"].median():.2f} MPa')
print(f'   Max stress  : {df["max_stress_all"].max():.2f} MPa')
print(f'   Fail threshold (quantile={STRESS_FAIL_THRESHOLD}): {threshold:.2f} MPa')
print(f'\n🏷️  Labels: {n_pass} PASS, {n_fail} FAIL ({100*n_fail/len(df):.1f}% fail rate)')


# ============================================================
# CELL 1.4  —  Only TWO lines changed (marked with # <-- CHANGED)
# ============================================================
# In the existing cell 1.4, find and replace these 3 lines:
#
#   OLD (lines to find):
#       # Determine label
#       stress_val = stress_data.get(name, 0.5)
#       label = 1.0 if stress_val >= STRESS_FAIL_THRESHOLD else 0.0
#
#   NEW (replace with):
#       # Determine label from real FEA data
#       label = label_data.get(name, 0.0)
#
# That's it! Everything else in cell 1.4 stays the same.
