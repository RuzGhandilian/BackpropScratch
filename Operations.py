import numpy as np


class ForwardOperations:
    @staticmethod
    def add(x1, x2):
        return x1 + x2

    @staticmethod
    def mul(x1, x2):
        return x1 * x2

    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class BackwardOperations:
    @staticmethod
    def add(dL_dy):
        return dL_dy, dL_dy

    @staticmethod
    def mul(dL_dy, x1, x2):
        return dL_dy * x2, dL_dy * x1

    @staticmethod
    def exp(dL_dy, x):
        return dL_dy * np.exp(x)

    @staticmethod
    def sigmoid(dL_dy, x):
        sigmoid_x = ForwardOperations.sigmoid(x)
        return dL_dy * sigmoid_x * (1 - sigmoid_x)
