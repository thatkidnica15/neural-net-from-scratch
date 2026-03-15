"""
Autograd Engine
===============
Tensor class with reverse-mode automatic differentiation.

The computational graph is built implicitly during forward operations.
Each Tensor stores:
  - .data: the actual numpy array
  - .grad: accumulated gradient (same shape as .data)
  - ._backward: a closure that computes local gradients
  - ._prev: set of parent Tensors in the graph

Calling .backward() on the loss Tensor traverses the graph in reverse
topological order, invoking each node's _backward function to propagate
gradients via the chain rule.

This is mathematically equivalent to PyTorch's autograd.

Author: Veronica Pilagov
"""

import numpy as np
from typing import Optional, Set, Callable


class Tensor:
    """
    A differentiable tensor with automatic gradient computation.
    
    Supports arithmetic operations (+, -, *, @, **) that build a
    computational graph for reverse-mode autodiff.
    """
    
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = np.array(data, dtype=np.float64) if not isinstance(data, np.ndarray) else data.astype(np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.requires_grad = requires_grad
        self._backward: Callable = lambda: None
        self._prev: Set['Tensor'] = set(_children)
        self._op = _op  # for debugging
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def T(self):
        """Transpose."""
        out = Tensor(self.data.T, (self,), 'T')
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Sum over broadcasted dimensions
            self_grad = out.grad
            other_grad = out.grad
            
            # Handle broadcasting: sum over axes that were broadcast
            while self_grad.ndim > self.data.ndim:
                self_grad = self_grad.sum(axis=0)
            for i, (s, o) in enumerate(zip(self.data.shape, self_grad.shape)):
                if s == 1 and o > 1:
                    self_grad = self_grad.sum(axis=i, keepdims=True)
            
            while other_grad.ndim > other.data.ndim:
                other_grad = other_grad.sum(axis=0)
            for i, (s, o) in enumerate(zip(other.data.shape, other_grad.shape)):
                if s == 1 and o > 1:
                    other_grad = other_grad.sum(axis=i, keepdims=True)
            
            self.grad += self_grad
            other.grad += other_grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        """Element-wise multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            self_grad = out.grad * other.data
            other_grad = out.grad * self.data
            
            while self_grad.ndim > self.data.ndim:
                self_grad = self_grad.sum(axis=0)
            for i, (s, o) in enumerate(zip(self.data.shape, self_grad.shape)):
                if s == 1 and o > 1:
                    self_grad = self_grad.sum(axis=i, keepdims=True)
                    
            while other_grad.ndim > other.data.ndim:
                other_grad = other_grad.sum(axis=0)
            for i, (s, o) in enumerate(zip(other.data.shape, other_grad.shape)):
                if s == 1 and o > 1:
                    other_grad = other_grad.sum(axis=i, keepdims=True)
            
            self.grad += self_grad
            other.grad += other_grad
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        """Matrix multiplication: C = A @ B, dA = dC @ B^T, dB = A^T @ dC."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, (self, other), '@')
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        
        out._backward = _backward
        return out
    
    def __pow__(self, exponent):
        """Power: d/dx(x^n) = n * x^(n-1)."""
        assert isinstance(exponent, (int, float))
        out = Tensor(self.data ** exponent, (self,), f'**{exponent}')
        
        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return (-self) + other
    
    def sum(self, axis=None, keepdims=False):
        """Summation with gradient propagation."""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')
        
        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.data.shape)
        
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        """Mean with gradient propagation."""
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)
    
    def reshape(self, *shape):
        """Reshape preserving gradient flow."""
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape')
        
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        
        out._backward = _backward
        return out
    
    def exp(self):
        """Exponential: d/dx(e^x) = e^x."""
        out = Tensor(np.exp(np.clip(self.data, -500, 500)), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def log(self):
        """Natural log: d/dx(ln(x)) = 1/x."""
        out = Tensor(np.log(np.clip(self.data, 1e-12, None)), (self,), 'log')
        
        def _backward():
            self.grad += (1.0 / np.clip(self.data, 1e-12, None)) * out.grad
        
        out._backward = _backward
        return out
    
    def clip(self, min_val, max_val):
        """Clip with straight-through gradient estimator."""
        out = Tensor(np.clip(self.data, min_val, max_val), (self,), 'clip')
        
        def _backward():
            mask = ((self.data >= min_val) & (self.data <= max_val)).astype(float)
            self.grad += mask * out.grad
        
        out._backward = _backward
        return out
    
    def maximum(self, val):
        """Element-wise maximum (used for ReLU): d/dx max(x,0) = 1 if x>0."""
        out = Tensor(np.maximum(self.data, val), (self,), 'max')
        
        def _backward():
            self.grad += (self.data > val).astype(float) * out.grad
        
        out._backward = _backward
        return out
    
    def backward(self):
        """
        Reverse-mode automatic differentiation.
        
        1. Build topological ordering of the computational graph
        2. Set gradient of this node (loss) to 1
        3. Traverse in reverse order, calling _backward at each node
        
        This computes dL/dx for every Tensor x in the graph.
        """
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        """Reset gradient to zero."""
        self.grad = np.zeros_like(self.data)
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data.flatten()[:4]}...)"
