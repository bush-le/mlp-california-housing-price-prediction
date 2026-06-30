import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, FEATURE_CLIP_RANGE, TARGET_SCALE_FACTOR, TARGET_COL, RANDOM_SEED
from src.preprocessing import StandardScaler
np.random.seed(RANDOM_SEED)

def scale_features(X_train, X_test, clip_range=(-5, 5), log_fn=print):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check if they are DataFrames
    if hasattr(X_train_scaled, 'values'): X_train_scaled = X_train_scaled.values
    if hasattr(X_test_scaled, 'values'): X_test_scaled = X_test_scaled.values
    
    X_train_scaled = np.clip(X_train_scaled, clip_range[0], clip_range[1])
    X_test_scaled = np.clip(X_test_scaled, clip_range[0], clip_range[1])
    log_fn(f"  Scaled features (StandardScaler, clip {clip_range})")
    return X_train_scaled, X_test_scaled, scaler

def scale_target(y_train, y_test, scale_factor, log_fn=print):
    y_train_scaled = y_train / scale_factor
    y_test_scaled = y_test / scale_factor
    log_fn(f"  Target scaled by / {scale_factor}")
    return y_train_scaled, y_test_scaled

def main():
    log_path = os.path.join(LOGS_DIR, '07_scaling.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')
    p("=" * 60)
    p("  STAGE 07 — FEATURE SCALING")
    p("=" * 60)
    p("  This stage is integrated into prepare_data.py pipeline.")
    log.close()

if __name__ == '__main__':
    main()