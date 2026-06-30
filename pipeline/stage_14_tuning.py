import os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOGS_DIR, PLOTS_DIR, EXPERIMENTS_DIR, EPOCHS, BATCH_SIZE, RANDOM_SEED, HIDDEN_LAYERS
from prepare_data import get_prepared_data
from src.model import MLP
from src.layers import Dense
from src.activations import ReLU, Linear
from src.losses import MSE
from src.optimizer import SGD
np.random.seed(RANDOM_SEED)

def build_and_train_mlp(X_train, y_train, X_test, y_test, lr):
    n_samples, n_features = X_train.shape
    model = MLP()
    model.add(Dense(n_features, HIDDEN_LAYERS[0]))
    model.add(ReLU())
    for i in range(len(HIDDEN_LAYERS) - 1):
        model.add(Dense(HIDDEN_LAYERS[i], HIDDEN_LAYERS[i+1]))
        model.add(ReLU())
    model.add(Dense(HIDDEN_LAYERS[-1], 1))
    model.add(Linear())
    model.compile(loss_function=MSE(), optimizer=SGD(learning_rate=lr))
    
    for epoch in range(10): # Shorter epochs for tuning
        indices = np.random.permutation(n_samples)
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]
        for i in range(0, n_samples, BATCH_SIZE):
            model.train_step(X_train_shuffled[i:i+BATCH_SIZE], y_train_shuffled[i:i+BATCH_SIZE])
            
    y_val_pred = model.predict(X_test)
    val_loss = model.loss_function.forward(y_val_pred, y_test)
    return val_loss

def main():
    log_path = os.path.join(LOGS_DIR, '14_hyperparameter_sweep.txt')
    log = open(log_path, 'w')
    def p(msg=''): print(msg); log.write(str(msg) + '\n')
    p("=" * 60)
    p("  STAGE 14 — HYPERPARAMETER TUNING")
    p("=" * 60)

    X_train, X_test, y_train, y_test, _ = get_prepared_data()
    
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    losses = []
    
    for lr in learning_rates:
        p(f"  Testing LR: {lr}")
        loss = build_and_train_mlp(X_train, y_train, X_test, y_test, lr)
        losses.append(loss)
        with open(os.path.join(EXPERIMENTS_DIR, f'experiment_lr_{lr}.txt'), 'w') as f:
            f.write(f"Learning Rate: {lr}\nValidation Loss: {loss}\n")
        p(f"    Val Loss: {loss:.5f}")
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.log10(learning_rates), losses, marker='o')
    ax.set_title('Learning Rate Tuning')
    ax.set_xlabel('Log10(Learning Rate)')
    ax.set_ylabel('Validation MSE')
    plot_path = os.path.join(PLOTS_DIR, '14_tuning_lr.png')
    fig.savefig(plot_path)
    plt.close(fig)
    p(f"\n  [✓] Tuning curve -> {plot_path}")
    log.close()

if __name__ == '__main__':
    main()