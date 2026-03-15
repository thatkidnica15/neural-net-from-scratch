"""
XOR Demo: Non-Linear Classification
====================================
The classic test: XOR cannot be solved by a linear model.
A single hidden layer with 2+ neurons can learn it.

Author: Veronica Pilagov
"""

import sys
import numpy as np
sys.path.insert(0, '..')

from minigrad import Tensor, Dense, ReLU, Sigmoid, BinaryCrossEntropy, Adam, Sequential

np.random.seed(42)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

# Build model
model = Sequential()
model.add(Dense(2, 8))
model.add(ReLU())
model.add(Dense(8, 1))
model.add(Sigmoid())
model.compile(optimiser_class=Adam, loss=BinaryCrossEntropy(), lr=0.05)

# Train (no validation split for 4 samples)
print("Training on XOR...")
for epoch in range(1, 501):
    xb = Tensor(X)
    yb = Tensor(y)
    pred = model.forward(xb)
    loss = model.loss_fn(pred, yb)
    model.optimiser.zero_grad()
    loss.backward()
    model.optimiser.step()
    
    if epoch % 100 == 0:
        preds = model.predict(X)
        acc = ((preds >= 0.5).astype(int) == y).mean()
        print(f"  Epoch {epoch} | Loss: {loss.data:.4f} | Accuracy: {acc:.2f}")

# Final predictions
print("\nFinal predictions:")
preds = model.predict(X)
for i in range(4):
    print(f"  Input: {X[i]} | Predicted: {preds[i][0]:.4f} | Target: {y[i][0]}")

print("\nXOR solved!" if ((preds >= 0.5).astype(int) == y).all() else "\nXOR not fully solved.")
