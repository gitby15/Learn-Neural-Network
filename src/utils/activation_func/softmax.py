import numpy as np


def _softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 减去最大值以防止溢出
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def _softmax_derivative(x):
    s = _softmax(x)
    return s * (1 - s)


def activation(x):
    return _softmax(x)


def derivative(x):
    return _softmax_derivative(x)
