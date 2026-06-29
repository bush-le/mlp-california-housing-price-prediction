"""
Stage 07 — Categorical Encoding
=================================
Reference: ML_PIPELINE_REFERENCE.md §7

ocean_proximity is the only categorical feature:
  • Categories: <1H OCEAN, INLAND, NEAR BAY, NEAR OCEAN, ISLAND
  • No natural order → One-Hot Encoding (§7.4)
  • Learn categories from TRAIN only to prevent data leakage

numpy implementation note: pd.get_dummies on train-learned categories.
Log → results/logs/07_encoding.txt
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, CATEGORICAL_COL, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def one_hot_encode(X_train_df, X_test_df, cat_col, log_fn=print):
    """
    One-hot encode a categorical column.
    Categories are learned from X_train only (no data leakage).

    Returns: X_train_encoded (numpy), X_test_encoded (numpy), known_categories
    """
    if cat_col not in X_train_df.columns:
        log_fn(f"  Column '{cat_col}' not found. Skipping encoding.")
        return X_train_df.values, X_test_df.values, []

    # Learn categories from train set only
    known_categories = sorted(X_train_df[cat_col].dropna().unique().tolist())
    log_fn(f"  Categorical column: {cat_col}")
    log_fn(f"  Categories learned from train: {known_categories}")

    # Apply same category set to both train and test
    X_train_df = X_train_df.copy()
    X_test_df = X_test_df.copy()
    X_train_df[cat_col] = pd.Categorical(X_train_df[cat_col],
                                          categories=known_categories)
    X_test_df[cat_col] = pd.Categorical(X_test_df[cat_col],
                                         categories=known_categories)

    # One-hot encode
    X_train_encoded = pd.get_dummies(X_train_df, columns=[cat_col], dtype=float)
    X_test_encoded = pd.get_dummies(X_test_df, columns=[cat_col], dtype=float)

    log_fn(f"  Columns after encoding: {X_train_encoded.shape[1]}")
    log_fn(f"  New columns: "
           f"{[c for c in X_train_encoded.columns if cat_col in c]}")

    return (X_train_encoded.values.astype(np.float64),
            X_test_encoded.values.astype(np.float64),
            known_categories)


def main():
    log_path = os.path.join(LOGS_DIR, '07_encoding.txt')
    log = open(log_path, 'w')

    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')

    p("=" * 60)
    p("  STAGE 07 — CATEGORICAL ENCODING")
    p("=" * 60)
    p(f"  Categorical feature: {CATEGORICAL_COL}")
    p(f"  Method: One-Hot Encoding (no natural order)")
    p(f"  Reason: §7.4 — equidistant categories, no false ordering")
    p(f"  This stage is integrated into main.py pipeline.")

    log.close()
    print(f"\n[✓] Log saved → {log_path}")


if __name__ == '__main__':
    main()
