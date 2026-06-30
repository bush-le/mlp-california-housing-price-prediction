import os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, PLOTS_DIR, MODELS_DIR, RANDOM_SEED
from prepare_data import get_prepared_data
from src.gmm import GMM
np.random.seed(RANDOM_SEED)

def main():
    log_path = os.path.join(LOGS_DIR, '13_gmm_training.txt')
    log = open(log_path, 'w')
    def p(msg=''): print(msg); log.write(str(msg) + '\n')
    p("=" * 60)
    p("  STAGE 13 — GMM TRAINING")
    p("=" * 60)

    X_train, X_test, _, _, _ = get_prepared_data()
    n_samples, n_features = X_train.shape
    
    components_range = [2, 3, 4]
    bics = []
    best_model = None
    best_bic = np.inf
    
    for k in components_range:
        p(f"\n  Training GMM with k={k} components...")
        model = GMM(n_components=k, max_iter=50, random_state=RANDOM_SEED)
        model.fit(X_train, log_fn=p)
        ll = model.log_likelihood_
        
        # Calculate BIC
        n_params = (k - 1) + k * n_features + k * n_features * (n_features + 1) / 2
        bic = n_params * np.log(n_samples) - 2 * ll
        bics.append(bic)
        p(f"  BIC for k={k}: {bic:.2f}")
        
        if bic < best_bic:
            best_bic = bic
            best_model = model

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(components_range, bics, marker='o')
    ax.set_title('BIC Score vs Number of Components')
    ax.set_xlabel('Components (k)')
    ax.set_ylabel('BIC Score')
    plot_path = os.path.join(PLOTS_DIR, '13_gmm_bic.png')
    fig.savefig(plot_path)
    plt.close(fig)
    p(f"\n  [✓] BIC curve -> {plot_path}")

    params = {'means': best_model.means_, 'covariances': best_model.covariances_, 'weights': best_model.weights_}
    params_path = os.path.join(MODELS_DIR, 'gmm_params.npy')
    np.save(params_path, params, allow_pickle=True)
    
    log.close()
    return best_model

if __name__ == '__main__':
    main()