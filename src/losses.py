import numpy as np

class MSE:
    def forward(self, y_pred, y_true):
        # Tính trung bình bình phương sai số
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_pred, y_true):
        # Đạo hàm của MSE theo y_pred: 2/N * (y_pred - y_true)
        return 2 * (y_pred - y_true) / y_true.size