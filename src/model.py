class MLP:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss_function, optimizer):
        self.loss_function = loss_function
        self.optimizer = optimizer

    def predict(self, input_data):
        # Lan truyền xuôi (Forward Propagation) qua từng lớp
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train_step(self, x_batch, y_batch):
        # 1. Forward
        output = self.predict(x_batch)
        
        # 2. Tính Loss (để in ra màn hình)
        loss = self.loss_function.forward(output, y_batch)
        
        # 3. Backward (Lan truyền ngược)
        grad = self.loss_function.backward(output, y_batch)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
        # 4. Update Weights (Tối ưu hóa)
        for layer in self.layers:
            self.optimizer.update(layer)
            
        return loss