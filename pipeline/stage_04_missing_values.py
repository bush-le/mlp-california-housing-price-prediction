"""
Stage 04 — Missing Value Handling
==================================
Reference: ML_PIPELINE_REFERENCE.md §4

Decision tree (§4.3):
  total_bedrooms — numerical, skewed → Median imputation
  All other numerical — no missing values
  ocean_proximity — categorical, no missing

Uses pure numpy/pandas. No sklearn.
Log → results/logs/04_imputation.txt
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DATA_PATH, LOGS_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def handle_missing_values(df, log_fn=print):
    """
    Impute missing values in-place and return the cleaned DataFrame.
    Strategy chosen per-feature based on distribution analysis (§4.3).
    """
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0]

    if len(cols_with_missing) == 0:
        log_fn("No missing values found.")
        return df

    log_fn(f"Columns with missing values:")
    for col, n in cols_with_missing.items():
        dtype = df[col].dtype
        pct = n / len(df) * 100

        if np.issubdtype(dtype, np.number):
            # Check skewness to decide mean vs median
            skew = df[col].dropna().skew()
            if abs(skew) < 0.5:
                strategy = 'mean'
                fill_val = df[col].mean()
            else:
                strategy = 'median'
                fill_val = df[col].median()

            df[col] = df[col].fillna(fill_val)
            log_fn(f"  {col:25s}  {n:5d} missing ({pct:.2f}%)  "
                   f"skew={skew:+.2f} → {strategy} imputation (fill={fill_val:.2f})")
        else:
            # Categorical → mode
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            log_fn(f"  {col:25s}  {n:5d} missing ({pct:.2f}%)  "
                   f"→ mode imputation (fill='{mode_val}')")

    return df


def main():
    df = pd.read_csv(RAW_DATA_PATH)

    log_path = os.path.join(LOGS_DIR, '04_imputation.txt')
    log = open(log_path, 'w')

    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')

    p("=" * 60)
    p("  STAGE 04 — MISSING VALUE HANDLING")
    p("=" * 60)
    p(f"Shape before: {df.shape}")

    df = handle_missing_values(df, log_fn=p)

    # Verify
    remaining = df.isnull().sum().sum()
    p(f"\nRemaining missing values: {remaining}")
    p(f"Shape after: {df.shape}")

    log.close()
    print(f"\n[✓] Log saved → {log_path}")
    return df


if __name__ == '__main__':
    main()
