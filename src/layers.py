import numpy as np

class Layer:
    def forward(self, input):
        pass
    def backward(self, output_gradient):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Khởi tạo trọng số ngẫu nhiên (He Initialization tốt cho ReLU)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        # Y = X.W + B
        return np.dot(self.input, self.weights) + self.biases

    def backward(self, output_gradient):
        # Tính toán Gradient để cập nhật trọng số
        self.dweights = np.dot(self.input.T, output_gradient)
        self.dbiases = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Gradient truyền ngược về lớp trước: dX = dY . W^T
        return np.dot(output_gradient, self.weights.T)