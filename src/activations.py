import numpy as np

class Activation:
    def forward(self, x):
        pass
    def backward(self, output_gradient):
        pass

class ReLU(Activation):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, output_gradient):
        # Đạo hàm ReLU: 1 nếu x > 0, ngược lại là 0
        return output_gradient * (self.input > 0)

class Sigmoid(Activation):
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, output_gradient):
        # Đạo hàm Sigmoid: s * (1 - s)
        return output_gradient * (self.output * (1 - self.output))

# Hàm Linear dùng cho lớp Output của bài toán hồi quy
class Linear(Activation):
    def forward(self, x):
        return x # Giữ nguyên giá trị
    
    def backward(self, output_gradient):
        return output_gradient # Đạo hàm của x là 1