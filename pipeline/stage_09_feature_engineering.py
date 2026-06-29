"""
Stage 09 — Feature Engineering
================================
Reference: ML_PIPELINE_REFERENCE.md §9

Create domain-specific features from the California Housing dataset:

1. rooms_per_household   = total_rooms / households
   → Per-household density is more predictive than raw counts.

2. bedrooms_per_room     = total_bedrooms / total_rooms
   → Ratio captures bedroom proportion; related to house size.

3. population_per_household = population / households
   → Household crowding correlates with housing prices.

These are interaction/ratio features (§9.4) that provide more
geometric signal than raw counts in the MLP feature space.

Plot → results/plots/09_engineered_*.png
Log  → results/logs/09_feature_engineering.txt
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, PLOTS_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def engineer_features(df, log_fn=print):
    """
    Add engineered features to the DataFrame.
    All operations are safe division with np.where to avoid div-by-zero.
    """
    df = df.copy()

    # 1. Rooms per household
    df['rooms_per_household'] = np.where(
        df['households'] > 0,
        df['total_rooms'] / df['households'],
        0.0
    )
    log_fn(f"  Created: rooms_per_household = total_rooms / households")

    # 2. Bedrooms per room
    df['bedrooms_per_room'] = np.where(
        df['total_rooms'] > 0,
        df['total_bedrooms'] / df['total_rooms'],
        0.0
    )
    log_fn(f"  Created: bedrooms_per_room = total_bedrooms / total_rooms")

    # 3. Population per household
    df['population_per_household'] = np.where(
        df['households'] > 0,
        df['population'] / df['households'],
        0.0
    )
    log_fn(f"  Created: population_per_household = population / households")

    return df


def plot_engineered_features(df, save_dir):
    """Plot distributions of the three engineered features."""
    eng_cols = ['rooms_per_household', 'bedrooms_per_room',
                'population_per_household']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for i, (col, color) in enumerate(zip(eng_cols, colors)):
        ax = axes[i]
        data = df[col].dropna()
        # Clip extreme values for visualization
        p99 = data.quantile(0.99)
        data_viz = data[data <= p99]
        ax.hist(data_viz, bins=50, color=color, edgecolor='white', alpha=0.8)
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_ylabel('Count')
        ax.text(0.95, 0.90, f'mean={data.mean():.2f}\nmed={data.median():.2f}',
                transform=ax.transAxes, ha='right', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Engineered Feature Distributions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, '09_engineered_distributions.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    from stage_05_outliers import main as run_stage_05
    df = run_stage_05()  # Returns cleaned DataFrame

    log_path = os.path.join(LOGS_DIR, '09_feature_engineering.txt')
    log = open(log_path, 'w')

    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')

    p("=" * 60)
    p("  STAGE 09 — FEATURE ENGINEERING")
    p("=" * 60)

    p(f"\n  Shape before: {df.shape}")
    df = engineer_features(df, log_fn=p)
    p(f"  Shape after:  {df.shape}")

    # Plot
    plot_path = plot_engineered_features(df, PLOTS_DIR)
    p(f"\n  [✓] Engineered feature distributions → {plot_path}")

    # Show correlations of new features with target
    p("\n  Correlation with median_house_value:")
    eng_cols = ['rooms_per_household', 'bedrooms_per_room',
                'population_per_household']
    for col in eng_cols:
        r = df[col].corr(df['median_house_value'])
        p(f"    {col:30s}  r = {r:+.4f}")

    log.close()
    print(f"\n[✓] Log saved → {log_path}")
    return df


if __name__ == '__main__':
    main()
