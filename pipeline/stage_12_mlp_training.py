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
    def p(msg=''): print(msg); log.write(str(msg) + '\n')
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
    p(f"\n  [✓] Loss curve -> {plot_path}")

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