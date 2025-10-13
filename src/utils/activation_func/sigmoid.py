import numpy as np


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _sigmoid_derivative(x):
    return _sigmoid(x) * (1 - _sigmoid(x))


def activation(x):
    return _sigmoid(x)


def derivative(x):
    return _sigmoid_derivative(x)
