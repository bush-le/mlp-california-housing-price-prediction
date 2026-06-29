"""
Helper script to execute the preprocessing pipeline and return fully prepared data.
Combines Stages 02-09.
"""
import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TARGET_COL, CATEGORICAL_COL, TEST_SIZE, TARGET_SCALE_FACTOR, FEATURE_CLIP_RANGE, RANDOM_SEED
from stage_02_data_loading import load_raw_data
from stage_04_missing_values import handle_missing_values
from stage_05_outliers import treat_outliers
from stage_09_feature_engineering import engineer_features
from stage_07_encoding import one_hot_encode
from stage_06_scaling import scale_features, scale_target

def get_prepared_data():
    # 1. Load Data
    df = load_raw_data()

    # 2. Handle missing values
    df = handle_missing_values(df, log_fn=lambda x: None)
    
    # 3. Handle outliers (remove censored)
    df = treat_outliers(df, log_fn=lambda x: None)
    
    # 4. Feature engineering
    df = engineer_features(df, log_fn=lambda x: None)
    
    # Split features and target
    X_df = df.drop(columns=[TARGET_COL])
    y_df = df[TARGET_COL]
    
    # Train test split (numpy style, consistent with config)
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(X_df))
    test_samples = int(len(X_df) * TEST_SIZE)
    
    train_idx = indices[test_samples:]
    test_idx = indices[:test_samples]
    
    X_train_df = X_df.iloc[train_idx]
    X_test_df = X_df.iloc[test_idx]
    y_train = y_df.iloc[train_idx].values.reshape(-1, 1)
    y_test = y_df.iloc[test_idx].values.reshape(-1, 1)
    
    # 5. Encoding
    X_train_encoded, X_test_encoded, _ = one_hot_encode(
        X_train_df, X_test_df, CATEGORICAL_COL, log_fn=lambda x: None)
    
    # 6. Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_encoded, X_test_encoded, clip_range=FEATURE_CLIP_RANGE, log_fn=lambda x: None)
        
    y_train_scaled, y_test_scaled = scale_target(
        y_train, y_test, TARGET_SCALE_FACTOR, log_fn=lambda x: None)
        
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler
