"""
Stage 08 — Class Imbalance Check
==================================
Reference: ML_PIPELINE_REFERENCE.md §8

This is a REGRESSION problem (predicting median_house_value).
Class imbalance handling (SMOTE, class weighting) applies to
CLASSIFICATION tasks only.

This stage documents the decision to skip and shows the target
distribution instead.

Plot → results/plots/08_target_distribution.png
Log  → results/logs/08_imbalance.txt
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (RAW_DATA_PATH, LOGS_DIR, PLOTS_DIR,
                    TARGET_COL, CAP_VALUE, RANDOM_SEED)

np.random.seed(RANDOM_SEED)


def main():
    df = pd.read_csv(RAW_DATA_PATH)

    log_path = os.path.join(LOGS_DIR, '08_imbalance.txt')
    log = open(log_path, 'w')

    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')

    p("=" * 60)
    p("  STAGE 08 — CLASS IMBALANCE CHECK")
    p("=" * 60)

    p("\n  Problem Type: Supervised REGRESSION")
    p("  Target: median_house_value (continuous)")
    p("  → Class imbalance handling (SMOTE, class weighting) is for")
    p("    CLASSIFICATION only. Not applicable here.")

    p("\n  However, the target distribution has a known issue:")
    n_capped = (df[TARGET_COL] >= CAP_VALUE).sum()
    p(f"  Censored values (>= ${CAP_VALUE:,.0f}): {n_capped} "
      f"({n_capped / len(df) * 100:.1f}%)")
    p(f"  These are handled in Stage 05 (outlier treatment).")

    # Plot target distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[TARGET_COL], bins=50, color='steelblue',
            edgecolor='white', alpha=0.8)
    ax.axvline(x=CAP_VALUE, color='red', linestyle='--', linewidth=2,
               label=f'Cap = ${CAP_VALUE:,.0f}')
    ax.set_xlabel('Median House Value ($)')
    ax.set_ylabel('Count')
    ax.set_title('Target Variable Distribution', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, '08_target_distribution.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    p(f"\n  [✓] Target distribution plot → {plot_path}")
    p("\n  Decision: SKIP (regression task, not classification)")

    log.close()
    print(f"\n[✓] Log saved → {log_path}")


if __name__ == '__main__':
    main()
