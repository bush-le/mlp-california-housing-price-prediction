"""
Stage 02 — Data Loading
=======================
Loads the raw California Housing CSV, prints shape / dtypes / head,
and logs everything to results/logs/02_data_loading.txt.

Reference: ML_PIPELINE_REFERENCE.md §3.1 (Global Overview)
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path so `config` and `src` are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RAW_DATA_PATH, LOGS_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def load_raw_data():
    """Load the raw housing CSV and return a DataFrame."""
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}\n"
            "Please place housing.csv in data/raw/"
        )
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def main():
    log_path = os.path.join(LOGS_DIR, '02_data_loading.txt')
    log = open(log_path, 'w')

    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')

    p("=" * 60)
    p("  STAGE 02 — DATA LOADING")
    p("=" * 60)

    df = load_raw_data()

    p(f"\nFile: {RAW_DATA_PATH}")
    p(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    p("\n--- dtypes ---")
    for col in df.columns:
        p(f"  {col:25s}  {str(df[col].dtype)}")

    p("\n--- Missing values ---")
    missing = df.isnull().sum()
    for col, n in missing.items():
        if n > 0:
            p(f"  {col}: {n} ({n / len(df) * 100:.2f}%)")
    if missing.sum() == 0:
        p("  None")

    p("\n--- First 5 rows ---")
    p(df.head().to_string())

    p("\n--- Descriptive statistics ---")
    p(df.describe().to_string())

    log.close()
    print(f"\n[✓] Log saved → {log_path}")
    return df


if __name__ == '__main__':
    main()
