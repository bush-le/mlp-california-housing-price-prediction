import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, CATEGORICAL_COL, RANDOM_SEED
np.random.seed(RANDOM_SEED)

def one_hot_encode(X_train_df, X_test_df, cat_col, log_fn=print):
    if cat_col not in X_train_df.columns:
        return X_train_df.values, X_test_df.values, []
    known_categories = sorted(X_train_df[cat_col].dropna().unique().tolist())
    log_fn(f"  Learned categories from train: {known_categories}")
    
    X_train_df = X_train_df.copy()
    X_test_df = X_test_df.copy()
    X_train_df[cat_col] = pd.Categorical(X_train_df[cat_col], categories=known_categories)
    X_test_df[cat_col] = pd.Categorical(X_test_df[cat_col], categories=known_categories)
    
    X_train_encoded = pd.get_dummies(X_train_df, columns=[cat_col], dtype=float)
    X_test_encoded = pd.get_dummies(X_test_df, columns=[cat_col], dtype=float)
    
    return X_train_encoded, X_test_encoded, known_categories

def main():
    log_path = os.path.join(LOGS_DIR, '08_encoding.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')
    p("=" * 60)
    p("  STAGE 08 — ENCODING")
    p("=" * 60)
    p("  This stage is integrated into prepare_data.py pipeline.")
    log.close()

if __name__ == '__main__':
    main()