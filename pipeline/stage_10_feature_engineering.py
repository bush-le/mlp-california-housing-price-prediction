import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, PLOTS_DIR, RANDOM_SEED, TARGET_COL
np.random.seed(RANDOM_SEED)

def engineer_features(df, log_fn=print):
    df = df.copy()
    df['rooms_per_household'] = np.where(df['households'] > 0, df['total_rooms'] / df['households'], 0.0)
    df['bedrooms_per_room'] = np.where(df['total_rooms'] > 0, df['total_bedrooms'] / df['total_rooms'], 0.0)
    df['population_per_household'] = np.where(df['households'] > 0, df['population'] / df['households'], 0.0)
    log_fn(f"  Created engineered features.")
    return df

def main():
    from stage_06_outliers import main as run_stage_06
    train_df, test_df = run_stage_06()
    log_path = os.path.join(LOGS_DIR, '10_feature_engineering.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')
    p("=" * 60)
    p("  STAGE 10 — FEATURE ENGINEERING")
    p("=" * 60)
    train_df = engineer_features(train_df, log_fn=p)
    test_df = engineer_features(test_df, log_fn=p)
    log.close()
    return train_df, test_df

if __name__ == '__main__':
    main()