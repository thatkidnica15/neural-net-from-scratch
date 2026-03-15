"""
Activation Functions
====================
Author: Veronica Pilagov
"""

import numpy as np
from .engine import Tensor


class ReLU:
    def __call__(self, x: Tensor) -> Tensor:
        return x.maximum(0)
    def parameters(self): return []

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def __call__(self, x: Tensor) -> Tensor:
        out = Tensor(np.where(x.data > 0, x.data, self.alpha * x.data), (x,), 'leakyrelu')
        _alpha = self.alpha
        def _backward():
            x.grad += np.where(x.data > 0, 1.0, _alpha) * out.grad
        out._backward = _backward
        return out
    def parameters(self): return []

class Sigmoid:
    def __call__(self, x: Tensor) -> Tensor:
        s = 1.0 / (1.0 + np.exp(-np.clip(x.data, -500, 500)))
        out = Tensor(s, (x,), 'sigmoid')
        def _backward():
            x.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out
    def parameters(self): return []

class Tanh:
    def __call__(self, x: Tensor) -> Tensor:
        t = np.tanh(x.data)
        out = Tensor(t, (x,), 'tanh')
        def _backward():
            x.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    def parameters(self): return []

class Softmax:
    def __call__(self, x: Tensor) -> Tensor:
        shifted = x.data - x.data.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        s = exp / exp.sum(axis=-1, keepdims=True)
        out = Tensor(s, (x,), 'softmax')
        def _backward():
            # Jacobian-vector product for softmax
            for i in range(x.data.shape[0]):
                si = s[i].reshape(-1, 1)
                jac = np.diagflat(si) - si @ si.T
                x.grad[i] += jac @ out.grad[i]
        out._backward = _backward
        return out
    def parameters(self): return []
