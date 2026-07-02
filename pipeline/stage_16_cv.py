import os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, METRICS_DIR, PLOTS_DIR, RANDOM_SEED, EPOCHS, BATCH_SIZE
from prepare_data import get_prepared_data
from stage_12_mlp_training import build_mlp
np.random.seed(RANDOM_SEED)

def main():
    log_path = os.path.join(LOGS_DIR, '16_cross_validation.txt')
    log = open(log_path, 'w')
    def p(msg=''): print(msg); log.write(str(msg) + '\n')
    p("=" * 60); p("  STAGE 16 — K-FOLD CROSS VALIDATION"); p("=" * 60)

    X_train, _, y_train, _, _ = get_prepared_data()
    n_samples = len(X_train)
    k = 3
    fold_size = n_samples // k
    indices = np.random.permutation(n_samples)
    losses = []
    
    for fold in range(k):
        val_idx = indices[fold*fold_size:(fold+1)*fold_size]
        train_idx = np.concatenate([indices[:fold*fold_size], indices[(fold+1)*fold_size:]])
        
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]
        
        model = build_mlp(X_tr.shape[1])
        for epoch in range(10): # Shorter for speed
            for i in range(0, len(X_tr), BATCH_SIZE):
                model.train_step(X_tr[i:i+BATCH_SIZE], y_tr[i:i+BATCH_SIZE])
                
        val_loss = model.loss_function.forward(model.predict(X_va), y_va)
        losses.append(val_loss)
        p(f"  Fold {fold+1}/{k} Validation Loss: {val_loss:.5f}")
        
    mu, sigma = np.mean(losses), np.std(losses)
    p(f"\n  CV Results (μ ± σ): {mu:.5f} ± {sigma:.5f}")
    
    with open(os.path.join(METRICS_DIR, 'cv_results_mu_sigma.txt'), 'w') as f:
        f.write(f"CV Loss: {mu} +- {sigma}\n")
        
    # Plot CV losses Boxplot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(losses, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax.set_title('K-Fold Cross Validation Losses (K=3) - Boxplot')
    ax.set_ylabel('Validation Loss (MSE scaled)')
    ax.set_xticks([1])
    ax.set_xticklabels(['Models'])
    fig.savefig(os.path.join(PLOTS_DIR, '16_cv_losses_boxplot.png'), bbox_inches='tight')
    plt.close(fig)
    
    # Plot CV losses Bar chart with error bar
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['CV Models'], [mu], yerr=[sigma], capsize=10, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.set_title('K-Fold Cross Validation Mean Loss (K=3) - Bar Chart')
    ax.set_ylabel('Mean Validation Loss (MSE scaled)')
    fig.savefig(os.path.join(PLOTS_DIR, '16_cv_losses_barchart.png'), bbox_inches='tight')
    plt.close(fig)
    
    log.close()

if __name__ == '__main__':
    main()