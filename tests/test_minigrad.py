"""
Tests for minigrad
==================
Includes numerical gradient verification to validate
that analytical gradients are correct.

Author: Veronica Pilagov
"""

import sys
import numpy as np
sys.path.insert(0, '..')

from minigrad import (
    Tensor, Dense, ReLU, Sigmoid, Softmax, Tanh,
    BinaryCrossEntropy, MeanSquaredError,
    Adam, SGD, Sequential
)


def numerical_gradient(f, x, eps=1e-5):
    """
    Compute numerical gradient via central differences.
    
    f'(x) ≈ [f(x+eps) - f(x-eps)] / (2*eps)
    
    This is O(eps^2) accurate, sufficient for gradient checking.
    """
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old_val = x.data[idx]
        
        x.data[idx] = old_val + eps
        fx_plus = f().data.copy()
        
        x.data[idx] = old_val - eps
        fx_minus = f().data.copy()
        
        grad[idx] = (fx_plus - fx_minus) / (2 * eps)
        x.data[idx] = old_val
        it.iternext()
    
    return grad


def test_add_gradient():
    a = Tensor(np.random.randn(3, 2))
    b = Tensor(np.random.randn(3, 2))
    
    def f():
        return (a + b).sum()
    
    result = f()
    result.backward()
    
    num_grad_a = numerical_gradient(f, a)
    num_grad_b = numerical_gradient(f, b)
    
    assert np.allclose(a.grad, num_grad_a, atol=1e-5), "Add gradient (a) failed"
    assert np.allclose(b.grad, num_grad_b, atol=1e-5), "Add gradient (b) failed"
    print("  PASS: test_add_gradient")


def test_matmul_gradient():
    a = Tensor(np.random.randn(4, 3))
    b = Tensor(np.random.randn(3, 2))
    
    def f():
        return (a @ b).sum()
    
    result = f()
    result.backward()
    
    num_grad_a = numerical_gradient(f, a)
    num_grad_b = numerical_gradient(f, b)
    
    assert np.allclose(a.grad, num_grad_a, atol=1e-5), "Matmul gradient (a) failed"
    assert np.allclose(b.grad, num_grad_b, atol=1e-5), "Matmul gradient (b) failed"
    print("  PASS: test_matmul_gradient")


def test_relu_gradient():
    x = Tensor(np.random.randn(5, 3))
    
    def f():
        return x.maximum(0).sum()
    
    result = f()
    result.backward()
    
    num_grad = numerical_gradient(f, x)
    assert np.allclose(x.grad, num_grad, atol=1e-5), "ReLU gradient failed"
    print("  PASS: test_relu_gradient")


def test_sigmoid_gradient():
    x = Tensor(np.random.randn(4, 2))
    sig = Sigmoid()
    
    def f():
        return sig(x).sum()
    
    result = f()
    x.grad = np.zeros_like(x.data)
    result.backward()
    
    num_grad = numerical_gradient(f, x)
    assert np.allclose(x.grad, num_grad, atol=1e-5), "Sigmoid gradient failed"
    print("  PASS: test_sigmoid_gradient")


def test_dense_gradient():
    np.random.seed(42)
    layer = Dense(3, 2)
    x = Tensor(np.random.randn(4, 3))
    
    def f():
        return layer(x).sum()
    
    result = f()
    layer.W.grad = np.zeros_like(layer.W.data)
    layer.b.grad = np.zeros_like(layer.b.data)
    x.grad = np.zeros_like(x.data)
    result.backward()
    
    num_grad_W = numerical_gradient(f, layer.W)
    num_grad_b = numerical_gradient(f, layer.b)
    
    assert np.allclose(layer.W.grad, num_grad_W, atol=1e-5), "Dense W gradient failed"
    assert np.allclose(layer.b.grad, num_grad_b, atol=1e-5), "Dense b gradient failed"
    print("  PASS: test_dense_gradient")


def test_bce_gradient():
    np.random.seed(42)
    pred = Tensor(np.random.uniform(0.1, 0.9, (4, 1)))
    target = Tensor(np.array([[0], [1], [1], [0]], dtype=np.float64))
    loss_fn = BinaryCrossEntropy()
    
    def f():
        return loss_fn(pred, target)
    
    result = f()
    pred.grad = np.zeros_like(pred.data)
    result.backward()
    
    num_grad = numerical_gradient(f, pred)
    assert np.allclose(pred.grad, num_grad, atol=1e-4), "BCE gradient failed"
    print("  PASS: test_bce_gradient")


def test_training_convergence():
    """Test that a simple network can learn a linearly separable problem."""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    
    model = Sequential()
    model.add(Dense(2, 16))
    model.add(ReLU())
    model.add(Dense(16, 1))
    model.add(Sigmoid())
    model.compile(optimiser_class=Adam, loss=BinaryCrossEntropy(), lr=0.01)
    
    history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=False)
    
    assert history['train_loss'][-1] < history['train_loss'][0], "Training loss did not decrease"
    assert history['train_acc'][-1] > 0.8, f"Accuracy too low: {history['train_acc'][-1]}"
    print("  PASS: test_training_convergence")


def test_softmax_gradient():
    x = Tensor(np.random.randn(3, 4))
    sm = Softmax()
    
    def f():
        return sm(x).sum()
    
    result = f()
    x.grad = np.zeros_like(x.data)
    result.backward()
    
    num_grad = numerical_gradient(f, x)
    assert np.allclose(x.grad, num_grad, atol=1e-4), "Softmax gradient failed"
    print("  PASS: test_softmax_gradient")


if __name__ == '__main__':
    print("Running minigrad tests...\n")
    
    tests = [
        test_add_gradient,
        test_matmul_gradient,
        test_relu_gradient,
        test_sigmoid_gradient,
        test_dense_gradient,
        test_bce_gradient,
        test_softmax_gradient,
        test_training_convergence,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__} - {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
