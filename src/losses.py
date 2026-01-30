import numpy as np

class MSE:
    def forward(self, y_pred, y_true):
        # Calculate Mean Squared Error
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_pred, y_true):
        # Derivative of MSE with respect to y_pred: 2/N * (y_pred - y_true)
        return 2 * (y_pred - y_true) / y_true.size