import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, PLOTS_DIR, NUM_FEATURES, CAP_VALUE, IQR_MULTIPLIER, RANDOM_SEED
np.random.seed(RANDOM_SEED)

def iqr_bounds(series, multiplier=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - multiplier * IQR, Q3 + multiplier * IQR

def detect_outliers(df, num_cols, multiplier=1.5):
    rows = []
    for col in num_cols:
        data = df[col].dropna()
        lower, upper = iqr_bounds(data, multiplier)
        n_low = int((data < lower).sum())
        n_high = int((data > upper).sum())
        rows.append({
            'Feature': col, 'Min': data.min(), 'Q1': data.quantile(0.25),
            'Median': data.median(), 'Q3': data.quantile(0.75), 'Max': data.max(),
            'Lower Fence': lower, 'Upper Fence': upper,
            'Total Outliers': n_low + n_high, '% Outliers': round((n_low + n_high) / len(data) * 100, 2)
        })
    return pd.DataFrame(rows).set_index('Feature')

def plot_boxplots(df, num_cols, summary, save_path):
    n = len(num_cols)
    ncols_grid = 3
    nrows_grid = (n + ncols_grid - 1) // ncols_grid
    fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(16, 4 * nrows_grid))
    axes = axes.flatten()
    colors = plt.cm.tab10.colors
    for i, col in enumerate(num_cols):
        ax = axes[i]
        bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True, boxprops=dict(facecolor=colors[i%10], alpha=0.6))
        n_out = int(summary.loc[col, 'Total Outliers'])
        pct = summary.loc[col, '% Outliers']
        ax.set_title(f'{col}\n({n_out:,} outliers, {pct:.1f}%)', fontsize=9)
        ax.set_xticks([])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Box Plots', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def treat_outliers(train_df, test_df, log_fn=print):
    n_before = len(train_df)
    train_df = train_df[train_df['median_house_value'] < CAP_VALUE].copy()
    n_removed_train = n_before - len(train_df)
    
    n_before_test = len(test_df)
    test_df = test_df[test_df['median_house_value'] < CAP_VALUE].copy()
    n_removed_test = n_before_test - len(test_df)
    
    log_fn(f"  Removed {n_removed_train} censored from train, {n_removed_test} from test (target >= {CAP_VALUE})")
    return train_df, test_df

def main():
    from stage_05_missing_values import main as run_stage_05
    train_df, test_df = run_stage_05()
    log_path = os.path.join(LOGS_DIR, '06_outliers.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')
    p("=" * 60)
    p("  STAGE 06 — OUTLIER DETECTION AND TREATMENT")
    p("=" * 60)
    
    num_cols = [c for c in NUM_FEATURES if c in train_df.columns]
    summary = detect_outliers(train_df, num_cols, IQR_MULTIPLIER) # Train only bounds
    
    before_path = os.path.join(PLOTS_DIR, '06_outliers_before.png')
    plot_boxplots(train_df, num_cols, summary, before_path)
    
    train_df, test_df = treat_outliers(train_df, test_df, log_fn=p)
    
    summary_after = detect_outliers(train_df, num_cols, IQR_MULTIPLIER)
    after_path = os.path.join(PLOTS_DIR, '06_outliers_after.png')
    plot_boxplots(train_df, num_cols, summary_after, after_path)
    
    log.close()
    return train_df, test_df

if __name__ == '__main__':
    main()