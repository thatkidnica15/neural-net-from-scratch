# minigrad

A deep learning framework built from scratch in NumPy. No PyTorch. No TensorFlow. Just linear algebra.

## Why Build This?

Frameworks abstract away the mathematics of learning. This library implements every component of a modern neural network from first principles, as a demonstration that I understand what happens between `model.fit()` and a trained network.

Every line of code maps to a specific equation. Every design decision is documented with the mathematical reasoning behind it.

## What It Implements

### Core Engine (`minigrad/engine.py`)
- **Tensor class** with automatic gradient tracking
- **Computational graph** construction during forward pass
- **Reverse-mode automatic differentiation** (backpropagation) via topological sort

### Layers (`minigrad/layers.py`)
- `Dense` — fully connected layer with He/Xavier initialisation
- `Conv2D` — 2D convolution via im2col (matrix multiplication, not nested loops)
- `Flatten` — reshape for CNN-to-dense transitions
- `BatchNorm` — batch normalisation with running statistics for inference
- `Dropout` — inverted dropout with configurable rate

### Activations (`minigrad/activations.py`)
- ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
- Each implemented with both forward and analytical backward pass

### Loss Functions (`minigrad/losses.py`)
- Binary cross-entropy
- Categorical cross-entropy
- Mean squared error

### Optimisers (`minigrad/optimisers.py`)
- SGD with momentum
- Adam (adaptive moment estimation)
- RMSProp

### Model API (`minigrad/model.py`)
- Sequential model with `.add()`, `.compile()`, `.fit()`, `.predict()`
- Training loop with mini-batch SGD, validation split, and history tracking

## Demo: MNIST Digit Classification

```bash
python demos/mnist_demo.py
```

Trains a CNN on MNIST handwritten digits using only this library:
- Architecture: Conv2D(1→8, 3×3) → ReLU → Conv2D(8→16, 3×3) → ReLU → Flatten → Dense(256) → ReLU → Dropout(0.3) → Dense(10) → Softmax
- Achieves ~95% test accuracy in ~10 epochs
- Generates training curves and sample predictions

## Demo: XOR Problem (Sanity Check)

```bash
python demos/xor_demo.py
```

The classic non-linearly separable problem. Proves the network can learn non-linear decision boundaries.

## Project Structure

```
minigrad/
├── minigrad/
│   ├── __init__.py
│   ├── engine.py          # Tensor with autograd
│   ├── layers.py          # Dense, Conv2D, BatchNorm, Dropout, Flatten
│   ├── activations.py     # ReLU, Sigmoid, Tanh, Softmax
│   ├── losses.py          # BCE, CCE, MSE
│   ├── optimisers.py      # SGD, Adam, RMSProp
│   └── model.py           # Sequential model API
├── demos/
│   ├── xor_demo.py        # XOR classification
│   └── mnist_demo.py      # MNIST CNN (requires sklearn for data)
├── tests/
│   └── test_minigrad.py   # Gradient checks and unit tests
└── README.md
```

## Technical Highlights

**Automatic differentiation**: The `Tensor` class builds a computational graph during forward operations. Calling `.backward()` traverses this graph in reverse topological order, computing gradients via the chain rule. This is the same approach used by PyTorch's autograd engine.

**im2col convolution**: Rather than implementing convolution as four nested loops (O(n⁶) for a batch), `Conv2D` uses the im2col trick to reshape input patches into columns, then computes all convolutions as a single matrix multiplication. This is how cuDNN implements convolutions on GPU.

**Numerical gradient checking**: The test suite includes numerical gradient verification (finite differences) to validate that every analytical gradient is correct to within 1e-5 relative error.

## Author

**Veronica Pilagov** — BSc Medical Innovation & Enterprise, UCL  
Built to demonstrate understanding of deep learning mathematics for Columbia MSAI application.

## License

MIT
