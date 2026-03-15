"""
Optimisers
==========
Parameter update rules for gradient descent variants.

Author: Veronica Pilagov
"""

import numpy as np
from typing import List
from .engine import Tensor


class SGD:
    """
    Stochastic Gradient Descent with momentum.
    
    v_t = momentum * v_{t-1} + lr * grad
    W = W - v_t
    
    Momentum accumulates velocity in persistent gradient directions,
    dampening oscillations and accelerating convergence.
    """
    def __init__(self, parameters: List[Tensor], lr: float = 0.01, momentum: float = 0.9):
        self.params = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]
    
    def step(self):
        for i, p in enumerate(self.params):
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * p.grad
            p.data -= self.velocities[i]
    
    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


class Adam:
    """
    Adam optimiser (Kingma & Ba, 2014).
    
    Combines momentum (first moment) with RMSProp (second moment)
    and applies bias correction for the initial time steps.
    
    m_t = beta1 * m_{t-1} + (1-beta1) * grad         (first moment)
    v_t = beta2 * v_{t-1} + (1-beta2) * grad^2        (second moment)
    m_hat = m_t / (1 - beta1^t)                        (bias correction)
    v_hat = v_t / (1 - beta2^t)
    W = W - lr * m_hat / (sqrt(v_hat) + eps)
    """
    def __init__(self, parameters: List[Tensor], lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


class RMSProp:
    """
    RMSProp (Hinton, unpublished).
    
    Adapts learning rate per-parameter using a moving average of
    squared gradients. Effective for non-stationary objectives.
    
    v_t = decay * v_{t-1} + (1-decay) * grad^2
    W = W - lr * grad / (sqrt(v_t) + eps)
    """
    def __init__(self, parameters: List[Tensor], lr: float = 0.001,
                 decay: float = 0.99, eps: float = 1e-8):
        self.params = parameters
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.v = [np.zeros_like(p.data) for p in self.params]
    
    def step(self):
        for i, p in enumerate(self.params):
            self.v[i] = self.decay * self.v[i] + (1 - self.decay) * (p.grad ** 2)
            p.data -= self.lr * p.grad / (np.sqrt(self.v[i]) + self.eps)
    
    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)
