"""
Stage 06 — Feature Scaling
===========================
Reference: ML_PIPELINE_REFERENCE.md §6

MLP (Neural Network) requires feature scaling (§6.2 — slow convergence,
unstable training without it). Using StandardScaler per §6.4:
  • Distribution is approximately Gaussian for most features → Standardization
  • No fixed known bounds → MinMax not appropriate
  • After scaling, clip extreme z-scores to [-5, 5] to handle rare
    one-hot categories (e.g., ISLAND with ~5 samples → z ≈ 73)

CRITICAL: Fit scaler on TRAIN only, transform on test. (§6.4 AI Agent Rule)

Log → results/logs/06_scaling.txt
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (LOGS_DIR, FEATURE_CLIP_RANGE, TARGET_SCALE_FACTOR,
                    TARGET_COL, RANDOM_SEED)

np.random.seed(RANDOM_SEED)


class NumpyStandardScaler:
    """
    StandardScaler implemented from scratch in numpy.
    x' = (x - μ) / σ     (§6.3)

    Fit on train data only. Transform applied identically to val/test.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Compute mean and std from training data."""
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Prevent division by zero for constant features
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        """Apply standardization: (X - mean) / std."""
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """Recover original values: X_scaled * std + mean."""
        X_scaled = np.asarray(X_scaled, dtype=np.float64)
        return X_scaled * self.std_ + self.mean_


def scale_features(X_train, X_test, clip_range=(-5, 5), log_fn=print):
    """
    Standardize features and clip extreme values.
    Returns: X_train_scaled, X_test_scaled, scaler
    """
    scaler = NumpyStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_fn(f"  Scaler fit on training data ({X_train.shape[0]} samples)")
    log_fn(f"  Mean (first 5): {scaler.mean_[:5]}")
    log_fn(f"  Std  (first 5): {scaler.std_[:5]}")

    # Verify
    log_fn(f"\n  After scaling (train):")
    log_fn(f"    mean ≈ {np.mean(X_train_scaled):.6f}  (should be ~0)")
    log_fn(f"    std  ≈ {np.std(X_train_scaled):.6f}   (should be ~1)")

    # Clip extreme z-scores
    X_train_scaled = np.clip(X_train_scaled, clip_range[0], clip_range[1])
    X_test_scaled = np.clip(X_test_scaled, clip_range[0], clip_range[1])

    log_fn(f"\n  Clipped to [{clip_range[0]}, {clip_range[1]}]")
    log_fn(f"    max(train) = {X_train_scaled.max():.2f}")
    log_fn(f"    min(train) = {X_train_scaled.min():.2f}")

    return X_train_scaled, X_test_scaled, scaler


def scale_target(y_train, y_test, scale_factor, log_fn=print):
    """
    Scale the target variable by dividing by scale_factor.
    Housing prices ~[15k, 500k] → dividing by 100k gives [0.15, 5.0].
    This prevents MSE loss from exploding and causing NaN gradients.
    """
    y_train_scaled = y_train / scale_factor
    y_test_scaled = y_test / scale_factor

    log_fn(f"\n  Target scaling:")
    log_fn(f"    Factor: ÷ {scale_factor:,.0f}")
    log_fn(f"    y_train range: [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")
    log_fn(f"    y_test  range: [{y_test_scaled.min():.4f}, {y_test_scaled.max():.4f}]")

    return y_train_scaled, y_test_scaled


def main():
    """Standalone run for testing — normally called from main.py."""
    log_path = os.path.join(LOGS_DIR, '06_scaling.txt')
    log = open(log_path, 'w')

    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')

    p("=" * 60)
    p("  STAGE 06 — FEATURE SCALING (standalone test)")
    p("=" * 60)
    p("  This stage is integrated into main.py pipeline.")
    p("  Run main.py for the full pipeline.")

    log.close()
    print(f"\n[✓] Log saved → {log_path}")


if __name__ == '__main__':
    main()
