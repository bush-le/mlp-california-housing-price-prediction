import os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, PLOTS_DIR, METRICS_DIR, TARGET_SCALE_FACTOR
from prepare_data import get_prepared_data

def evaluate_mlp(model, X_test_scaled, y_test_scaled, log_fn):
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = y_pred_scaled * TARGET_SCALE_FACTOR
    y_test = y_test_scaled * TARGET_SCALE_FACTOR
    
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
    
    log_fn("  MLP Regression Metrics:")
    log_fn(f"    RMSE: ${rmse:,.2f}\n    MAE:  ${mae:,.2f}\n    R²:   {r2:.4f}")
    
    threshold = 250000.0
    y_test_bin = (y_test > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)
    
    tp = np.sum((y_pred_bin == 1) & (y_test_bin == 1))
    tn = np.sum((y_pred_bin == 0) & (y_test_bin == 0))
    fp = np.sum((y_pred_bin == 1) & (y_test_bin == 0))
    fn = np.sum((y_pred_bin == 0) & (y_test_bin == 1))
    
    acc = (tp + tn) / len(y_test_bin)
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else 0
    
    log_fn(f"\n  MLP Classification (>{threshold}):\n    Acc: {acc:.4f}\n    Prec: {prec:.4f}\n    Rec: {rec:.4f}\n    F1: {f1:.4f}")
    
    with open(os.path.join(METRICS_DIR, 'mlp_final_metrics.txt'), 'w') as f:
        f.write(f"RMSE:{rmse}\nMAE:{mae}\nR2:{r2}\nAcc:{acc}\nF1:{f1}\n")
        
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow([[tn, fp], [fn, tp]], cmap='Blues')
    for (i, j), z in np.ndenumerate([[tn, fp], [fn, tp]]):
        ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred <= 250k', 'Pred > 250k'])
    ax.set_yticklabels(['True <= 250k', 'True > 250k'])
    cm_path = os.path.join(PLOTS_DIR, '15_confusion_matrix.png')
    fig.savefig(cm_path)
    plt.close(fig)

def main(mlp_model=None):
    if mlp_model is None: return
    log_path = os.path.join(LOGS_DIR, '15_evaluation.txt')
    log = open(log_path, 'w')
    def p(msg=''): print(msg); log.write(str(msg) + '\n')
    p("=" * 60); p("  STAGE 15 — EVALUATION METRICS"); p("=" * 60)
    _, X_test_scaled, _, y_test_scaled, _ = get_prepared_data()
    evaluate_mlp(mlp_model, X_test_scaled, y_test_scaled, p)
    log.close()

if __name__ == '__main__':
    main()
