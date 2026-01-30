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
        # Derivative of ReLU: 1 if x > 0, else 0
        return output_gradient * (self.input > 0)

class Sigmoid(Activation):
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, output_gradient):
        # Derivative of Sigmoid: s * (1 - s)
        return output_gradient * (self.output * (1 - self.output))

# Linear function used for the output layer in regression problems
class Linear(Activation):
    def forward(self, x):
        return x # Returns the input value unchanged

    def backward(self, output_gradient):
        return output_gradient # Derivative of x is 1