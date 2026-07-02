import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, PLOTS_DIR, RANDOM_SEED
np.random.seed(RANDOM_SEED)

def handle_missing_values(train_df, test_df, log_fn=print):
    train_df_before = train_df.copy()
    train_df = train_df.copy()
    test_df = test_df.copy()
    missing = train_df.isnull().sum()
    cols_with_missing = missing[missing > 0]
    
    if len(cols_with_missing) == 0:
        log_fn("No missing values found in train set.")
        return train_df, test_df

    log_fn(f"Columns with missing values (based on train set):")
    for col, n in cols_with_missing.items():
        dtype = train_df[col].dtype
        pct = n / len(train_df) * 100
        if np.issubdtype(dtype, np.number):
            skew = train_df[col].dropna().skew()
            if abs(skew) < 0.5:
                strategy = 'mean'
                fill_val = train_df[col].mean()
            else:
                strategy = 'median'
                fill_val = train_df[col].median()
            log_fn(f"  {col:25s}  {n:5d} missing ({pct:.2f}%)  skew={skew:+.2f} → {strategy} (fill={fill_val:.2f})")
            
            # Plot before/after for numerical columns
            fig, ax = plt.subplots(figsize=(8, 5))
            train_df_before[col].dropna().plot(kind='kde', ax=ax, label='Before Imputation (drop NA)', color='red')
            train_df[col] = train_df[col].fillna(fill_val)
            train_df[col].plot(kind='kde', ax=ax, label=f'After Imputation (fill {strategy})', color='green', linestyle='--')
            ax.set_title(f'Distribution of {col} Before and After Imputation')
            ax.legend()
            fig.savefig(os.path.join(PLOTS_DIR, f'05_imputation_{col}.png'), bbox_inches='tight')
            plt.close(fig)
            
        else:
            mode_val = train_df[col].mode()[0]
            fill_val = mode_val
            log_fn(f"  {col:25s}  {n:5d} missing ({pct:.2f}%)  → mode (fill='{mode_val}')")
            train_df[col] = train_df[col].fillna(fill_val)
            
        test_df[col] = test_df[col].fillna(fill_val)
        
    return train_df, test_df

def main():
    from stage_04_split import main as run_split
    train_df, test_df = run_split()
    log_path = os.path.join(LOGS_DIR, '05_imputation.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')
    p("=" * 60)
    p("  STAGE 05 — MISSING VALUE HANDLING")
    p("=" * 60)
    train_df, test_df = handle_missing_values(train_df, test_df, log_fn=p)
    log.close()
    return train_df, test_df

if __name__ == '__main__':
    main()