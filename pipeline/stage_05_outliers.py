"""
Stage 05 — Outlier Detection and Treatment
============================================
Reference: ML_PIPELINE_REFERENCE.md §5

Detection: IQR method (§5.2) — Q1 - 1.5*IQR, Q3 + 1.5*IQR
Treatment decisions (from original notebook analysis):
  • total_rooms, total_bedrooms, population, households:
    → Keep. Genuine large blocks. Controlled via StandardScaler + clip.
  • median_income:
    → Keep. Most important feature per EDA.
  • median_house_value >= $500,001:
    → REMOVE. These are censored (price-capped) data points.
    → The dataset uses $500,001 as a sentinel value — the true price is unknown.
    → Keeping them corrupts the MLP regression target.

Plots   → results/plots/05_outliers_*.png
Log     → results/logs/05_outliers.txt
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (RAW_DATA_PATH, LOGS_DIR, PLOTS_DIR,
                    NUM_FEATURES, CAP_VALUE, IQR_MULTIPLIER, RANDOM_SEED)

np.random.seed(RANDOM_SEED)


def iqr_bounds(series, multiplier=1.5):
    """Return (lower_bound, upper_bound) using the IQR rule."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - multiplier * IQR, Q3 + multiplier * IQR


def detect_outliers(df, num_cols, multiplier=1.5):
    """Return a summary DataFrame with outlier stats per column."""
    rows = []
    for col in num_cols:
        data = df[col].dropna()
        lower, upper = iqr_bounds(data, multiplier)
        n_low  = int((data < lower).sum())
        n_high = int((data > upper).sum())
        rows.append({
            'Feature': col,
            'Min': data.min(), 'Q1': data.quantile(0.25),
            'Median': data.median(), 'Q3': data.quantile(0.75),
            'Max': data.max(),
            'Lower Fence': lower, 'Upper Fence': upper,
            'Outlier Low': n_low, 'Outlier High': n_high,
            'Total Outliers': n_low + n_high,
            '% Outliers': round((n_low + n_high) / len(data) * 100, 2),
        })
    return pd.DataFrame(rows).set_index('Feature')


def plot_boxplots(df, num_cols, summary, save_path):
    """Box plots for every numerical feature, with outlier counts."""
    n = len(num_cols)
    ncols_grid = 3
    nrows_grid = (n + ncols_grid - 1) // ncols_grid

    fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(16, 4 * nrows_grid))
    axes = axes.flatten()
    colors = plt.cm.tab10.colors

    for i, col in enumerate(num_cols):
        ax = axes[i]
        bp = ax.boxplot(
            df[col].dropna(), vert=True, patch_artist=True,
            boxprops=dict(facecolor=colors[i % 10], alpha=0.6),
            medianprops=dict(color='black', linewidth=2),
            flierprops=dict(marker='o', markerfacecolor='red',
                            markersize=2.5, alpha=0.4, linestyle='none'),
            whiskerprops=dict(linestyle='--'),
        )
        n_out = int(summary.loc[col, 'Total Outliers'])
        pct = summary.loc[col, '% Outliers']
        ax.set_title(f'{col}\n({n_out:,} outliers, {pct:.1f}%)', fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f'{x:,.0f}'))
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.set_xticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Box Plots — Outlier Analysis', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def treat_outliers(df, log_fn=print):
    """
    Apply treatment decisions:
      1. Remove censored target values (>= $500,001)
      2. Keep numerical feature outliers (handled downstream by scaler + clip)
    """
    n_before = len(df)

    # Remove censored houses
    df = df[df['median_house_value'] < CAP_VALUE].copy()
    n_removed = n_before - len(df)

    log_fn(f"\n  Treatment: Removed {n_removed} censored houses "
           f"(median_house_value >= ${CAP_VALUE:,.0f})")
    log_fn(f"  Reason:    Dataset uses ${CAP_VALUE:,.0f} as sentinel — "
           f"true price unknown.")
    log_fn(f"  Shape after removal: {df.shape}")

    log_fn(f"\n  Numerical feature outliers (total_rooms, population, etc.):")
    log_fn(f"    → Kept intact. These represent genuine large housing blocks.")
    log_fn(f"    → Will be controlled by StandardScaler + clip to [-5, 5].")

    return df


def main():
    df = pd.read_csv(RAW_DATA_PATH)
    # Apply missing value handling first (required before outlier analysis)
    from stage_04_missing_values import handle_missing_values
    df = handle_missing_values(df, log_fn=lambda x: None)  # silent

    log_path = os.path.join(LOGS_DIR, '05_outliers.txt')
    log = open(log_path, 'w')

    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')

    p("=" * 60)
    p("  STAGE 05 — OUTLIER DETECTION AND TREATMENT")
    p("=" * 60)

    # ── Detection ──────────────────────────────────────────────
    num_cols = [c for c in NUM_FEATURES if c in df.columns]
    summary = detect_outliers(df, num_cols, IQR_MULTIPLIER)

    p(f"\nSamples: {len(df):,}")
    p(f"\n{'Feature':25s} {'Total':>8s} {'%':>7s}  Decision")
    p("-" * 65)
    for col in num_cols:
        row = summary.loc[col]
        decision = "REMOVE (censored)" if col == 'median_house_value' else "KEEP (genuine)"
        p(f"  {col:25s} {int(row['Total Outliers']):>6,}  "
          f"{row['% Outliers']:>6.2f}%  {decision}")

    # ── Box plots BEFORE treatment ─────────────────────────────
    before_path = os.path.join(PLOTS_DIR, '05_outliers_before.png')
    plot_boxplots(df, num_cols, summary, before_path)
    p(f"\n  [✓] Before-treatment box plots → {before_path}")

    # ── Treatment ──────────────────────────────────────────────
    df = treat_outliers(df, log_fn=p)

    # ── Box plots AFTER treatment ──────────────────────────────
    summary_after = detect_outliers(df, num_cols, IQR_MULTIPLIER)
    after_path = os.path.join(PLOTS_DIR, '05_outliers_after.png')
    plot_boxplots(df, num_cols, summary_after, after_path)
    p(f"\n  [✓] After-treatment box plots → {after_path}")

    log.close()
    print(f"\n[✓] Log saved → {log_path}")
    return df


if __name__ == '__main__':
    main()
