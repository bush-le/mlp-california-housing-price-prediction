import sys, os, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, METRICS_DIR, PLOTS_DIR, TARGET_SCALE_FACTOR
from prepare_data import get_prepared_data

def main():
    log_path = os.path.join(LOGS_DIR, '11_baseline.txt')
    log = open(log_path, 'w')
    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')
    p("=" * 60)
    p("  STAGE 11 — BASELINE MODEL")
    p("=" * 60)
    
    X_train, X_test, y_train, y_test, _ = get_prepared_data()
    mean_pred = np.mean(y_train)
    y_pred = np.full_like(y_test, fill_value=mean_pred)
    
    y_pred_unscaled = y_pred * TARGET_SCALE_FACTOR
    y_test_unscaled = y_test * TARGET_SCALE_FACTOR
    
    rmse = np.sqrt(np.mean((y_test_unscaled - y_pred_unscaled)**2))
    p(f"  Baseline (Mean Prediction) RMSE: ${rmse:,.2f}")
    
    with open(os.path.join(METRICS_DIR, 'baseline_metrics.txt'), 'w') as f:
        f.write(f"RMSE: {rmse}\n")
        
    # Plot Baseline Predictions
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test_unscaled, y_pred_unscaled, alpha=0.5, color='orange', edgecolors='k')
    ax.plot([y_test_unscaled.min(), y_test_unscaled.max()], [y_test_unscaled.min(), y_test_unscaled.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Prices')
    ax.set_ylabel('Baseline Predicted Prices (Mean)')
    ax.set_title('Baseline Model: Actual vs Predicted')
    ax.legend()
    fig.savefig(os.path.join(PLOTS_DIR, '11_baseline_pred.png'), bbox_inches='tight')
    plt.close(fig)

    log.close()

if __name__ == '__main__':
    main()