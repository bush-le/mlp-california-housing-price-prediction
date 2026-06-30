import os

base_dir = "/home/bush/Desktop/mlp-california-housing-price-prediction"

GMM_PY = """
import numpy as np

class GMM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
    def _mvn_pdf(self, X, mean, cov):
        n_features = X.shape[1]
        cov += np.eye(n_features) * 1e-6  # regularization for numerical stability
        diff = X - mean
        inv_cov = np.linalg.inv(cov)
        exponent = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        det = np.linalg.det(cov)
        return (1.0 / np.sqrt((2 * np.pi) ** n_features * det)) * np.exp(-0.5 * exponent)
        
    def fit(self, X, log_fn=None):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices]
        self.covariances_ = np.array([np.cov(X.T) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        log_likelihoods = []
        for i in range(self.max_iter):
            # E-step
            responsibilities = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                responsibilities[:, k] = self.weights_[k] * self._mvn_pdf(X, self.means_[k], self.covariances_[k])
                
            log_likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))
            log_likelihoods.append(log_likelihood)
            
            responsibilities = responsibilities / np.sum(responsibilities, axis=1, keepdims=True)
            
            # M-step
            N_k = np.sum(responsibilities, axis=0)
            for k in range(self.n_components):
                self.means_[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / N_k[k]
                diff = X - self.means_[k]
                self.covariances_[k] = np.dot(responsibilities[:, k] * diff.T, diff) / N_k[k]
                self.weights_[k] = N_k[k] / n_samples
                
            if log_fn:
                log_fn(f"  EM Iteration {i+1}: log-likelihood = {log_likelihood:.4f}")
                
            if i > 0 and np.abs(log_likelihood - log_likelihoods[-2]) < self.tol:
                if log_fn:
                    log_fn(f"  Converged after {i+1} iterations.")
                break
                
        self.log_likelihood_ = log_likelihoods[-1]
        self.history_ = log_likelihoods
        return self
        
    def score(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * self._mvn_pdf(X, self.means_[k], self.covariances_[k])
        return np.sum(np.log(np.sum(responsibilities, axis=1)))
"""

S12 = """
import os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, PLOTS_DIR, MODELS_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED, HIDDEN_LAYERS
from prepare_data import get_prepared_data
from src.model import MLP
from src.layers import Dense
from src.activations import ReLU, Linear
from src.losses import MSE
from src.optimizer import SGD
np.random.seed(RANDOM_SEED)

def build_mlp(input_size):
    model = MLP()
    model.add(Dense(input_size, HIDDEN_LAYERS[0]))
    model.add(ReLU())
    for i in range(len(HIDDEN_LAYERS) - 1):
        model.add(Dense(HIDDEN_LAYERS[i], HIDDEN_LAYERS[i+1]))
        model.add(ReLU())
    model.add(Dense(HIDDEN_LAYERS[-1], 1))
    model.add(Linear())
    model.compile(loss_function=MSE(), optimizer=SGD(learning_rate=LEARNING_RATE))
    return model

def main():
    log_path = os.path.join(LOGS_DIR, '12_mlp_training.txt')
    log = open(log_path, 'w')
    def p(msg=''): print(msg); log.write(str(msg) + '\\n')
    p("=" * 60)
    p("  STAGE 12 — MLP TRAINING")
    p("=" * 60)

    X_train, X_test, y_train, y_test, _ = get_prepared_data()
    n_samples, n_features = X_train.shape
    model = build_mlp(n_features)
    
    train_losses, val_losses = [], []
    for epoch in range(EPOCHS):
        indices = np.random.permutation(n_samples)
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]
        epoch_loss = 0
        n_batches = 0
        for i in range(0, n_samples, BATCH_SIZE):
            X_batch = X_train_shuffled[i:i+BATCH_SIZE]
            y_batch = y_train_shuffled[i:i+BATCH_SIZE]
            epoch_loss += model.train_step(X_batch, y_batch)
            n_batches += 1
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        
        y_val_pred = model.predict(X_test)
        val_loss = model.loss_function.forward(y_val_pred, y_test)
        val_losses.append(val_loss)
        if (epoch + 1) % 10 == 0:
            p(f"  Epoch {epoch+1:03d}/{EPOCHS} - Train: {avg_train_loss:.5f} - Val: {val_loss:.5f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Train', color='blue')
    ax.plot(val_losses, label='Validation', color='orange')
    ax.set_title('MLP Training Curve')
    ax.legend()
    plot_path = os.path.join(PLOTS_DIR, '12_mlp_loss_curve.png')
    fig.savefig(plot_path)
    plt.close(fig)
    p(f"\\n  [✓] Loss curve -> {plot_path}")

    weights = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            weights.append({'weights': layer.weights, 'biases': layer.biases})
    weights_path = os.path.join(MODELS_DIR, 'mlp_weights.npy')
    np.save(weights_path, weights, allow_pickle=True)
    
    log.close()
    return model

if __name__ == '__main__':
    main()
"""

S13 = """
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
    def p(msg=''): print(msg); log.write(str(msg) + '\\n')
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
        p(f"\\n  Training GMM with k={k} components...")
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
    p(f"\\n  [✓] BIC curve -> {plot_path}")

    params = {'means': best_model.means_, 'covariances': best_model.covariances_, 'weights': best_model.weights_}
    params_path = os.path.join(MODELS_DIR, 'gmm_params.npy')
    np.save(params_path, params, allow_pickle=True)
    
    log.close()
    return best_model

if __name__ == '__main__':
    main()
"""

with open(os.path.join(base_dir, 'src', 'gmm.py'), 'w') as f:
    f.write(GMM_PY.strip())
with open(os.path.join(base_dir, 'pipeline', 'stage_12_mlp_training.py'), 'w') as f:
    f.write(S12.strip())
with open(os.path.join(base_dir, 'pipeline', 'stage_13_gmm_training.py'), 'w') as f:
    f.write(S13.strip())
