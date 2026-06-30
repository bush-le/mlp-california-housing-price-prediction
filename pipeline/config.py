"""
config.py — Central configuration for the California Housing MLP pipeline.

All magic numbers, file paths, and hyperparameters live here.
Every pipeline stage imports this file instead of hard-coding values.
"""

import os
import numpy as np

# ============================================================
# REPRODUCIBILITY
# ============================================================
RANDOM_SEED = 42

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'housing.csv')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Results — enforced by REFACTOR_PROMPTS.md
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results')
LOGS_DIR     = os.path.join(RESULTS_DIR, 'logs')
PLOTS_DIR    = os.path.join(RESULTS_DIR, 'plots')
METRICS_DIR  = os.path.join(RESULTS_DIR, 'metrics')
MODELS_DIR   = os.path.join(RESULTS_DIR, 'models')
EXPERIMENTS_DIR = os.path.join(RESULTS_DIR, 'experiments')

# ============================================================
# DATASET
# ============================================================
TARGET_COL = 'median_house_value'
CATEGORICAL_COL = 'ocean_proximity'

# Capped / censored houses in this dataset use $500,001 as sentinel
CAP_VALUE = 500_001.0

# Target is scaled by this factor so loss stays in a numerically stable range
TARGET_SCALE_FACTOR = 100_000.0

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
TEST_SIZE = 0.2

# ============================================================
# OUTLIER TREATMENT
# ============================================================
IQR_MULTIPLIER = 1.5

# After StandardScaler, clip extreme z-scores (rare one-hot categories
# like ISLAND with only ~5 samples produce z ≈ 73 after scaling)
FEATURE_CLIP_RANGE = (-5.0, 5.0)

# ============================================================
# MLP ARCHITECTURE
# ============================================================
HIDDEN_LAYERS = [64, 32, 16]   # neurons per hidden layer
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 16

# ============================================================
# NUMERICAL FEATURE NAMES (for outlier / EDA analysis)
# ============================================================
NUM_FEATURES = [
    'longitude', 'latitude', 'housing_median_age',
    'total_rooms', 'total_bedrooms', 'population',
    'households', 'median_income', 'median_house_value',
]

# ============================================================
# HELPER — create all output directories at import time
# ============================================================
for _d in [PROCESSED_DIR, LOGS_DIR, PLOTS_DIR, METRICS_DIR, MODELS_DIR, EXPERIMENTS_DIR]:
    os.makedirs(_d, exist_ok=True)
