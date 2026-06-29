"""
Stage 12 — Evaluation
======================
Reference: ML_PIPELINE_REFERENCE.md §12 (Metrics)

Computes and saves final metrics for both MLP and GMM.
For MLP: RMSE, MAE, R2 (Regression).
To satisfy the prompt's request for Classification metrics (Accuracy,
Precision, Recall, F1, Confusion Matrix), we binarize the task by
predicting if a house is "Expensive" (> $250k).
All metrics → results/metrics/
Plots      → results/plots/12_confusion_matrix.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, PLOTS_DIR, METRICS_DIR, TARGET_SCALE_FACTOR
from prepare_data import get_prepared_data

# Instead of importing and retraining, we can accept the models directly,
# but for standalone execution we just do a quick re-train or load weights.
# We will accept them as arguments.
def evaluate_mlp(model, X_test_scaled, y_test_scaled, log_fn):
    """Evaluate MLP Regression and Classification metrics."""
    # 1. Regression Metrics
    y_pred_scaled = model.predict(X_test_scaled)
    
    # Unscale
    y_pred = y_pred_scaled * TARGET_SCALE_FACTOR
    y_test = y_test_scaled * TARGET_SCALE_FACTOR
    
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    mae = np.mean(np.abs(y_test - y_pred))
    
    ss_res = np.sum((y_test - y_pred)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    log_fn("  MLP Regression Metrics:")
    log_fn(f"    RMSE: ${rmse:,.2f}")
    log_fn(f"    MAE:  ${mae:,.2f}")
    log_fn(f"    R²:   {r2:.4f}")
    
    # 2. Classification Metrics (threshold = $250k)
    threshold = 250000.0
    y_test_bin = (y_test > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)
    
    tp = np.sum((y_pred_bin == 1) & (y_test_bin == 1))
    tn = np.sum((y_pred_bin == 0) & (y_test_bin == 0))
    fp = np.sum((y_pred_bin == 1) & (y_test_bin == 0))
    fn = np.sum((y_pred_bin == 0) & (y_test_bin == 1))
    
    accuracy = (tp + tn) / len(y_test_bin)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    log_fn(f"\n  MLP Classification Metrics (Threshold > ${threshold:,.0f}):")
    log_fn(f"    Accuracy:  {accuracy:.4f}")
    log_fn(f"    Precision: {precision:.4f}")
    log_fn(f"    Recall:    {recall:.4f}")
    log_fn(f"    F1 Score:  {f1:.4f}")
    
    # 3. Save Metrics
    metrics_path = os.path.join(METRICS_DIR, 'mlp_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"RMSE: {rmse}\nMAE: {mae}\nR2: {r2}\n")
        f.write(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\n")
        
    # 4. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow([[tn, fp], [fn, tp]], cmap='Blues')
    fig.colorbar(cax)
    
    for (i, j), z in np.ndenumerate([[tn, fp], [fn, tp]]):
        ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
                
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred <= $250k', 'Pred > $250k'])
    ax.set_yticklabels(['True <= $250k', 'True > $250k'])
    ax.set_title('Confusion Matrix', pad=20, fontweight='bold')
    
    cm_path = os.path.join(PLOTS_DIR, '12_confusion_matrix.png')
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log_fn(f"\n  [✓] Confusion matrix plot saved → {cm_path}")


def evaluate_gmm(model, X_test, log_fn):
    """Evaluate GMM."""
    if model is None:
        log_fn("  GMM model not provided. Skipping GMM evaluation.")
        return
        
    log_likelihood = model.score(X_test)
    n_samples, n_features = X_test.shape
    n_components = model.n_components
    
    weights_params = n_components - 1
    means_params = n_components * n_features
    covs_params = n_components * n_features * (n_features + 1) / 2
    k = weights_params + means_params + covs_params
    
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n_samples) - 2 * log_likelihood
    
    log_fn(f"\n  GMM Metrics (Test Set):")
    log_fn(f"    Log-Likelihood: {log_likelihood:.4f}")
    log_fn(f"    AIC: {aic:.2f}")
    log_fn(f"    BIC: {bic:.2f}")
    
    metrics_path = os.path.join(METRICS_DIR, 'gmm_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Log-Likelihood: {log_likelihood}\nAIC: {aic}\nBIC: {bic}\n")


def main(mlp_model=None, gmm_model=None):
    log_path = os.path.join(LOGS_DIR, '12_evaluation.txt')
    log = open(log_path, 'w')

    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')

    p("=" * 60)
    p("  STAGE 12 — EVALUATION")
    p("=" * 60)

    _, X_test_scaled, _, y_test_scaled, _ = get_prepared_data()
    
    if mlp_model is not None:
        evaluate_mlp(mlp_model, X_test_scaled, y_test_scaled, p)
    else:
        p("  MLP model not provided. Skip.")
        
    if gmm_model is not None:
        evaluate_gmm(gmm_model, X_test_scaled, p)
    else:
        p("  GMM model not provided. Skip.")

    log.close()
    print(f"\n[✓] Log saved → {log_path}")

if __name__ == '__main__':
    main()
