import numpy as np
from Operations import ForwardOperations, BackwardOperations


class LinearRegression:
    def __init__(self):
        self.weights = np.random.randn(1)
        self.bias = np.random.randn(1)

    def forward(self, x):
        return ForwardOperations.add(ForwardOperations.mul(x, self.weights), self.bias)

    def backward(self, dL_dy, x):
        dL_dw, dL_db = BackwardOperations.mul(dL_dy, x, self.weights)
        return dL_dw, dL_db, dL_dy * np.ones_like(self.bias)

    def update_parameters(self, dL_dw, dL_db, learning_rate):
        self.weights -= learning_rate * dL_dw.mean()
        self.bias -= learning_rate * dL_db.mean()
