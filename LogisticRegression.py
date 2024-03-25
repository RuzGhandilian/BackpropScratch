import numpy as np
from Operations import ForwardOperations, BackwardOperations


class LogisticRegression:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def forward(self, x):
        linear = np.dot(x, self.weights) + self.bias
        return ForwardOperations.sigmoid(linear)

    def backward(self, predictions, targets, x):
        error = predictions - targets
        dL_dw = np.dot(x.T, error)
        dL_db = np.sum(error)
        return dL_dw, dL_db

    def update_parameters(self, dL_dw, dL_db, learning_rate):
        self.weights -= learning_rate * dL_dw
        self.bias -= learning_rate * dL_db
