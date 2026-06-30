import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TARGET_COL, CATEGORICAL_COL, TARGET_SCALE_FACTOR, FEATURE_CLIP_RANGE
from stage_04_split import split_data
from stage_05_missing_values import handle_missing_values
from stage_06_outliers import treat_outliers
from stage_10_feature_engineering import engineer_features
from stage_08_encoding import one_hot_encode
from stage_07_scaling import scale_features, scale_target

def get_prepared_data():
    from stage_02_data_loading import load_raw_data
    df = load_raw_data()
    # 4. Split FIRST
    train_df, test_df = split_data(df, log_fn=lambda x: None)
    # 5. Missing values
    train_df, test_df = handle_missing_values(train_df, test_df, log_fn=lambda x: None)
    # 6. Outliers
    train_df, test_df = treat_outliers(train_df, test_df, log_fn=lambda x: None)
    # 10. Engineering
    train_df = engineer_features(train_df, log_fn=lambda x: None)
    test_df = engineer_features(test_df, log_fn=lambda x: None)
    
    # 8. Encoding
    X_train_df = train_df.drop(columns=[TARGET_COL])
    X_test_df = test_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].values.reshape(-1, 1)
    y_test = test_df[TARGET_COL].values.reshape(-1, 1)
    
    X_train_enc, X_test_enc, _ = one_hot_encode(X_train_df, X_test_df, CATEGORICAL_COL, log_fn=lambda x: None)
    
    # 7. Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_enc, X_test_enc, clip_range=FEATURE_CLIP_RANGE, log_fn=lambda x: None)
    y_train_scaled, y_test_scaled = scale_target(y_train, y_test, TARGET_SCALE_FACTOR, log_fn=lambda x: None)
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler