"""
Stage 10 — MLP Training
=======================
Reference: ML_PIPELINE_REFERENCE.md §10

Train the MLP from scratch using the prepared data.
Logs every epoch to results/logs/10_mlp_training.txt
Saves loss curve to results/plots/10_mlp_loss_curve.png
Saves final weights to results/models/mlp_weights.npy
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    """Builds MLP architecture dynamically based on config."""
    model = MLP()
    
    # Input -> First Hidden
    model.add(Dense(input_size, HIDDEN_LAYERS[0]))
    model.add(ReLU())
    
    # Hidden -> Hidden
    for i in range(len(HIDDEN_LAYERS) - 1):
        model.add(Dense(HIDDEN_LAYERS[i], HIDDEN_LAYERS[i+1]))
        model.add(ReLU())
        
    # Last Hidden -> Output
    model.add(Dense(HIDDEN_LAYERS[-1], 1))
    model.add(Linear())
    
    model.compile(loss_function=MSE(), optimizer=SGD(learning_rate=LEARNING_RATE))
    return model


def main():
    log_path = os.path.join(LOGS_DIR, '10_mlp_training.txt')
    log = open(log_path, 'w')

    def p(msg=''):
        print(msg)
        log.write(str(msg) + '\n')

    p("=" * 60)
    p("  STAGE 10 — MLP TRAINING")
    p("=" * 60)

    # 1. Get Data
    p("  Loading and preparing data...")
    X_train, X_test, y_train, y_test, _ = get_prepared_data()
    n_samples, n_features = X_train.shape
    p(f"  Train shape: {X_train.shape}")
    
    # 2. Build Model
    model = build_mlp(n_features)
    p(f"  Architecture: {n_features} -> {HIDDEN_LAYERS} -> 1")
    p(f"  Optimizer: SGD (lr={LEARNING_RATE})")
    p(f"  Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
    
    # 3. Train
    train_losses = []
    val_losses = []
    
    p("\n  Starting training...")
    for epoch in range(EPOCHS):
        # Shuffle batches
        indices = np.random.permutation(n_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, BATCH_SIZE):
            X_batch = X_train_shuffled[i:i+BATCH_SIZE]
            y_batch = y_train_shuffled[i:i+BATCH_SIZE]
            
            # Forward + Backward pass
            batch_loss = model.train_step(X_batch, y_batch)
            epoch_loss += batch_loss
            n_batches += 1
            
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

        # Validation pass
        y_val_pred = model.predict(X_test)
        val_loss = model.loss_function.forward(y_val_pred, y_test)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            p(f"  Epoch {epoch+1:03d}/{EPOCHS} - Train Loss: {avg_train_loss:.5f} - Val Loss: {val_loss:.5f}")

    # 4. Plot Loss Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Train Loss (MSE scaled)', color='blue')
    ax.plot(val_losses, label='Validation Loss (MSE scaled)', color='orange')
    ax.set_title('MLP Training Curve', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, '10_mlp_loss_curve.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    p(f"\n  [✓] Loss curve saved → {plot_path}")

    # 5. Save Weights
    weights = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            weights.append({'weights': layer.weights, 'biases': layer.biases})
            
    weights_path = os.path.join(MODELS_DIR, 'mlp_weights.npy')
    np.save(weights_path, weights, allow_pickle=True)
    p(f"  [✓] Weights saved → {weights_path}")

    log.close()
    print(f"\n[✓] Log saved → {log_path}")
    return model


if __name__ == '__main__':
    main()
