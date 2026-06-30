import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DATA_PATH, LOGS_DIR, PLOTS_DIR, TEST_SIZE, RANDOM_SEED, TARGET_COL
from src.preprocessing import train_test_split
np.random.seed(RANDOM_SEED)

def split_data(df, log_fn=print):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train
    
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test
    
    log_fn(f"  Train shape: {train_df.shape}")
    log_fn(f"  Test shape:  {test_df.shape}")
    
    return train_df, test_df

def main():
    df = pd.read_csv(RAW_DATA_PATH)
    log_path = os.path.join(LOGS_DIR, '04_split.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')
    p("=" * 60)
    p("  STAGE 04 — TRAIN/TEST SPLIT (LEAKAGE BOUNDARY)")
    p("=" * 60)
    train_df, test_df = split_data(df, log_fn=p)
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(train_df[TARGET_COL].dropna(), bins=50, alpha=0.5, label='Train', density=True)
    ax.hist(test_df[TARGET_COL].dropna(), bins=50, alpha=0.5, label='Test', density=True)
    ax.set_title('Target Distribution (Train vs Test)')
    ax.legend()
    plot_path = os.path.join(PLOTS_DIR, '04_split_balance.png')
    fig.savefig(plot_path)
    plt.close(fig)
    p(f"\n  [✓] Split distribution plot saved -> {plot_path}")
    log.close()
    return train_df, test_df

if __name__ == '__main__':
    main()