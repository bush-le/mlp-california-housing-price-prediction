"""
Stage 03 — Exploratory Data Analysis (EDA)
==========================================
Performs the three EDA lenses from ML_PIPELINE_REFERENCE.md §3:
  Lens 1 — Global overview  (shape, types, missing, ranges)
  Lens 2 — Univariate       (histograms for every numerical feature)
  Lens 3 — Multivariate     (correlation heatmap, scatter on target)

All figures → results/plots/03_eda_*.png
Summary    → results/logs/03_eda_summary.txt
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # no GUI — save to file only
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (RAW_DATA_PATH, LOGS_DIR, PLOTS_DIR,
                    NUM_FEATURES, TARGET_COL, CATEGORICAL_COL, RANDOM_SEED)

np.random.seed(RANDOM_SEED)


def main():
    # ── Load data ──────────────────────────────────────────────
    df = pd.read_csv(RAW_DATA_PATH)

    log_path = os.path.join(LOGS_DIR, '03_eda_summary.txt')
    log = open(log_path, 'w')

    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')

    p("=" * 60)
    p("  STAGE 03 — EXPLORATORY DATA ANALYSIS")
    p("=" * 60)

    # ────────────────────────────────────────────────────────────
    # LENS 1: Global Overview
    # ────────────────────────────────────────────────────────────
    p("\n--- Lens 1: Global Overview ---")
    p(f"Samples : {df.shape[0]}")
    p(f"Features: {df.shape[1]}")

    p("\nColumn types:")
    for col in df.columns:
        p(f"  {col:25s}  {str(df[col].dtype):10s}  "
          f"missing={df[col].isnull().sum()}")

    p("\nValue ranges (numerical):")
    desc = df.describe().T
    for col in desc.index:
        p(f"  {col:25s}  min={desc.loc[col,'min']:12.1f}  "
          f"max={desc.loc[col,'max']:12.1f}  "
          f"mean={desc.loc[col,'mean']:12.1f}  "
          f"std={desc.loc[col,'std']:12.1f}")

    # ────────────────────────────────────────────────────────────
    # LENS 2: Univariate Distributions  (histograms)
    # ────────────────────────────────────────────────────────────
    p("\n--- Lens 2: Univariate Distributions ---")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(num_cols)
    ncols_grid = 3
    nrows_grid = (n + ncols_grid - 1) // ncols_grid

    fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(16, 4 * nrows_grid))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_ylabel('Count')
        skew = data.skew()
        ax.text(0.95, 0.90, f'skew={skew:.2f}', transform=ax.transAxes,
                ha='right', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        p(f"  {col:25s}  skew={skew:+.2f}  "
          f"median={data.median():.2f}  mean={data.mean():.2f}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Univariate Distributions — All Numerical Features',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    hist_path = os.path.join(PLOTS_DIR, '03_eda_histograms.png')
    fig.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    p(f"\n  [✓] Histograms saved → {hist_path}")

    # ── Categorical bar chart ──────────────────────────────────
    if CATEGORICAL_COL in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        counts = df[CATEGORICAL_COL].value_counts()
        ax.barh(counts.index, counts.values, color='teal', edgecolor='white')
        ax.set_xlabel('Count')
        ax.set_title(f'Distribution of {CATEGORICAL_COL}', fontweight='bold')
        plt.tight_layout()
        cat_path = os.path.join(PLOTS_DIR, '03_eda_ocean_proximity.png')
        fig.savefig(cat_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        p(f"  [✓] Categorical bar chart saved → {cat_path}")

    # ────────────────────────────────────────────────────────────
    # LENS 3: Multivariate — Correlation Heatmap
    # ────────────────────────────────────────────────────────────
    p("\n--- Lens 3: Multivariate Relationships ---")

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    # annotate
    for i_r in range(len(corr)):
        for j_c in range(len(corr)):
            ax.text(j_c, i_r, f'{corr.values[i_r, j_c]:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if abs(corr.values[i_r, j_c]) > 0.6 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Feature Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    corr_path = os.path.join(PLOTS_DIR, '03_eda_correlation.png')
    fig.savefig(corr_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    p(f"\nCorrelation with {TARGET_COL}:")
    target_corr = corr[TARGET_COL].sort_values(ascending=False)
    for feat, r in target_corr.items():
        p(f"  {feat:25s}  r = {r:+.4f}")

    p(f"\n  [✓] Correlation heatmap saved → {corr_path}")

    # ── Geographic scatter (California map) ────────────────────
    if 'longitude' in df.columns and 'latitude' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 7))
        sc = ax.scatter(df['longitude'], df['latitude'], alpha=0.35,
                        c=df[TARGET_COL], cmap='jet',
                        s=df['population'] / 100)
        fig.colorbar(sc, ax=ax, label='Median House Value ($)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('California Housing Price Map', fontweight='bold')
        plt.tight_layout()
        map_path = os.path.join(PLOTS_DIR, '03_eda_geo_map.png')
        fig.savefig(map_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        p(f"  [✓] Geographic map saved → {map_path}")

    log.close()
    print(f"\n[✓] EDA summary saved → {log_path}")


if __name__ == '__main__':
    main()
