import numpy as np
from typing import Union
import tensorflow as tf

Array = Union[float, np.ndarray]

class Relu:
    def forward(self, x):
        self.x = np.atleast_2d(x)
        return np.maximum(0, self.x)
    
    def backward(self, grad_output):
        grad_output = np.atleast_2d(grad_output)
        return grad_output * (self.x > 0)

class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.logits = None
        self.probs = None
        self.target = None

    def forward(self, x: Array, target: Array) -> float:
        self.logits = np.atleast_2d(x)
        if target.ndim == 1:
            one_hot = np.zeros((target.size, x.shape[1]))
            one_hot[np.arange(target.size), target] = 1
            self.target = one_hot
        else:
            self.target = target

        shifted_logits = self.logits - np.max(self.logits, axis=1, keepdims=True)
        exps = np.exp(shifted_logits)
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)

        eps = 1e-10
        log_probs = np.log(np.clip(self.probs, eps, 1 - eps))
        loss = -np.sum(log_probs * self.target) / self.logits.shape[0]
        return loss

    def backward(self):
        return (self.probs - self.target) / self.logits.shape[0]

class Dense:
    def __init__(self, input_dim: int, output_dim: int):
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.w = np.random.uniform(-limit, limit, (output_dim, input_dim))
        self.b = np.zeros((1, output_dim))
        self.input = None
        self.dw = None
        self.db = None

    def forward(self, x: Array) -> Array:
        self.input = np.atleast_2d(x)
        return np.dot(self.input, self.w.T) + self.b

    def backward(self, grad_output: Array) -> Array:
        grad_output = np.atleast_2d(grad_output)
        self.dw = np.dot(grad_output.T, self.input)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.w)
        return grad_input

    def update(self, lr: float = 0.01):
        self.w -= lr * self.dw
        self.b -= lr * self.db

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def fit(self, x, y, epochs, loss_fn, lr=0.01, batch = 64):
        for epoch in range(epochs):
            shuffle = np.random.permutation(x.shape[0])
            
            x_shuffle = x[shuffle]
            y_shuffle = y[shuffle]
            
            total_loss = 0
            correct = 0
            
            for i in range(0, x.shape[0], batch):
                
                x_tran = x_shuffle[i:i+batch]
                y_tran = y_shuffle[i:i+batch]
                output = self.forward(x_tran)
                loss = loss_fn.forward(output, y_tran)
                grad = loss_fn.backward()
                self.backward(grad)

                for layer in self.layers:
                    if hasattr(layer, 'update'):
                        layer.update(lr)

                total_loss += loss * x_tran.shape[0]
                preds = np.argmax(output, axis=1)
                labels = y_tran if y_tran.ndim == 1 else np.argmax(y_tran, axis=1)
                correct += np.sum(preds == labels)

            avg_loss = total_loss / x.shape[0]
            acc = correct / x.shape[0]
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

# Initialize model
model = Model()
model.add(Dense(784, 128))
model.add(Relu())
model.add(Dense(128, 128))
model.add(Relu())
model.add(Dense(128, 64))
model.add(Relu())
model.add(Dense(64, 10))

# Train model
loss = SoftmaxCrossEntropyLoss()
model.fit(x_train, y_train, epochs=20, loss_fn=loss, lr=0.1)
# import numpy as np
# a = np.array([[1, 2],
#               [3, 4],
#               [5, 6],
#               [7, 8],
#               [9, 10],
#               [11, 12],
#               [13, 14],
#               [15, 16]])
# b = np.random.permutation(8)
# print(a[b])

