"""
Neural Network Layers
=====================
Implementations of common layer types using the Tensor autograd engine.

Author: Veronica Pilagov
"""

import numpy as np
from .engine import Tensor


class Dense:
    """
    Fully connected layer: output = input @ W + b
    
    Weight initialisation:
    - He init for ReLU: W ~ N(0, sqrt(2/fan_in))
    - Xavier init for sigmoid/tanh: W ~ N(0, sqrt(2/(fan_in+fan_out)))
    """
    
    def __init__(self, in_features: int, out_features: int, init: str = 'he'):
        if init == 'he':
            scale = np.sqrt(2.0 / in_features)
        else:  # xavier
            scale = np.sqrt(2.0 / (in_features + out_features))
        
        self.W = Tensor(np.random.randn(in_features, out_features) * scale)
        self.b = Tensor(np.zeros((1, out_features)))
    
    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b
    
    def parameters(self):
        return [self.W, self.b]


class Conv2D:
    """
    2D Convolution using the im2col trick.
    
    Instead of nested loops, im2col unrolls input patches into columns
    of a matrix, then computes all convolutions as a single matmul.
    This is how production frameworks implement convolution.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 1 for grayscale, 3 for RGB).
    out_channels : int
        Number of filters (output channels).
    kernel_size : int
        Size of the square convolution kernel.
    stride : int
        Stride of the convolution.
    padding : int
        Zero-padding added to input borders.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_c = in_channels
        self.out_c = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He initialisation
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale)
        self.b = Tensor(np.zeros((out_channels, 1)))
    
    def _im2col(self, x, h_out, w_out):
        """
        Convert input patches to column matrix.
        
        For each output position, extract the corresponding input
        patch and stack as a column. This converts convolution into
        a single matrix multiplication.
        """
        batch, c, h, w = x.shape
        cols = np.zeros((batch, c * self.k * self.k, h_out * w_out))
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                patch = x[:, :, h_start:h_start+self.k, w_start:w_start+self.k]
                cols[:, :, i * w_out + j] = patch.reshape(batch, -1)
        
        return cols
    
    def __call__(self, x: Tensor) -> Tensor:
        batch, c, h, w = x.data.shape
        
        # Apply padding
        if self.padding > 0:
            padded = np.pad(x.data, ((0,0), (0,0), 
                           (self.padding, self.padding), 
                           (self.padding, self.padding)), mode='constant')
        else:
            padded = x.data
        
        h_pad, w_pad = padded.shape[2], padded.shape[3]
        h_out = (h_pad - self.k) // self.stride + 1
        w_out = (w_pad - self.k) // self.stride + 1
        
        # im2col
        cols = self._im2col(padded, h_out, w_out)
        
        # Reshape filters to 2D: (out_c, in_c*k*k)
        W_flat = self.W.data.reshape(self.out_c, -1)
        
        # Convolution as matmul: (out_c, in_c*k*k) @ (in_c*k*k, h_out*w_out)
        out_data = np.zeros((batch, self.out_c, h_out * w_out))
        for b in range(batch):
            out_data[b] = W_flat @ cols[b] + self.b.data
        
        out_data = out_data.reshape(batch, self.out_c, h_out, w_out)
        out = Tensor(out_data, (x, self.W, self.b), 'conv2d')
        
        # Store for backward
        _cols = cols
        _x_shape = x.data.shape
        _padded = padded
        
        def _backward():
            W_flat_bw = self.W.data.reshape(self.out_c, -1)
            dout = out.grad.reshape(batch, self.out_c, -1)
            
            # Gradient w.r.t. weights
            dW = np.zeros_like(self.W.data.reshape(self.out_c, -1))
            for b in range(batch):
                dW += dout[b] @ _cols[b].T
            self.W.grad += dW.reshape(self.W.data.shape)
            
            # Gradient w.r.t. bias
            self.b.grad += dout.sum(axis=(0, 2)).reshape(self.b.data.shape)
            
            # Gradient w.r.t. input (col2im)
            dx_padded = np.zeros_like(_padded)
            for b in range(batch):
                dcols = W_flat_bw.T @ dout[b]
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        patch = dcols[:, i * w_out + j].reshape(c, self.k, self.k)
                        dx_padded[b, :, h_start:h_start+self.k, w_start:w_start+self.k] += patch
            
            if self.padding > 0:
                x.grad += dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
            else:
                x.grad += dx_padded
        
        out._backward = _backward
        return out
    
    def parameters(self):
        return [self.W, self.b]


class Flatten:
    """Flatten spatial dimensions for CNN-to-dense transitions."""
    
    def __call__(self, x: Tensor) -> Tensor:
        batch = x.data.shape[0]
        flat = x.data.reshape(batch, -1)
        out = Tensor(flat, (x,), 'flatten')
        
        original_shape = x.data.shape
        def _backward():
            x.grad += out.grad.reshape(original_shape)
        out._backward = _backward
        return out
    
    def parameters(self):
        return []


class BatchNorm:
    """
    Batch Normalisation (Ioffe & Szegedy, 2015).
    
    Normalises layer inputs to zero mean, unit variance per feature,
    then applies learnable scale (gamma) and shift (beta).
    
    During training: uses batch statistics.
    During inference: uses exponential moving average of training statistics.
    """
    
    def __init__(self, n_features: int, momentum: float = 0.1):
        self.gamma = Tensor(np.ones((1, n_features)))
        self.beta = Tensor(np.zeros((1, n_features)))
        self.running_mean = np.zeros((1, n_features))
        self.running_var = np.ones((1, n_features))
        self.momentum = momentum
        self.training = True
    
    def __call__(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.data.mean(axis=0, keepdims=True)
            var = x.data.var(axis=0, keepdims=True)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        eps = 1e-5
        x_norm_data = (x.data - mean) / np.sqrt(var + eps)
        out_data = self.gamma.data * x_norm_data + self.beta.data
        out = Tensor(out_data, (x, self.gamma, self.beta), 'batchnorm')
        
        _x_norm = x_norm_data
        _std_inv = 1.0 / np.sqrt(var + eps)
        _m = x.data.shape[0]
        
        def _backward():
            dx_norm = out.grad * self.gamma.data
            self.gamma.grad += (out.grad * _x_norm).sum(axis=0, keepdims=True)
            self.beta.grad += out.grad.sum(axis=0, keepdims=True)
            
            dvar = (dx_norm * (x.data - mean) * -0.5 * (_std_inv ** 3)).sum(axis=0, keepdims=True)
            dmean = (dx_norm * -_std_inv).sum(axis=0, keepdims=True) + dvar * (-2.0 / _m) * (x.data - mean).sum(axis=0, keepdims=True)
            x.grad += dx_norm * _std_inv + dvar * 2.0 * (x.data - mean) / _m + dmean / _m
        
        out._backward = _backward
        return out
    
    def parameters(self):
        return [self.gamma, self.beta]


class Dropout:
    """
    Inverted Dropout.
    
    During training: randomly zero out neurons with probability p,
    then scale remaining activations by 1/(1-p).
    During inference: identity (no dropout).
    
    The inverted scaling means no modification is needed at test time.
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self.training = True
    
    def __call__(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        
        mask = (np.random.random(x.data.shape) > self.p).astype(float)
        scale = 1.0 / (1.0 - self.p)
        out_data = x.data * mask * scale
        out = Tensor(out_data, (x,), 'dropout')
        
        def _backward():
            x.grad += out.grad * mask * scale
        out._backward = _backward
        return out
    
    def parameters(self):
        return []
