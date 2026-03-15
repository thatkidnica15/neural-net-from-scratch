"""
Sequential Model
================
High-level API for building and training neural networks.

Author: Veronica Pilagov
"""

import numpy as np
from typing import List
from .engine import Tensor


class Sequential:
    """
    Sequential neural network model.
    
    Usage:
        model = Sequential()
        model.add(Dense(784, 128))
        model.add(ReLU())
        model.add(Dense(128, 10))
        model.add(Softmax())
        model.compile(optimiser=Adam, loss=CategoricalCrossEntropy, lr=0.001)
        history = model.fit(X_train, y_train, epochs=10, batch_size=32)
    """
    
    def __init__(self):
        self.layers = []
        self.optimiser = None
        self.loss_fn = None
    
    def add(self, layer):
        self.layers.append(layer)
        return self
    
    def compile(self, optimiser_class, loss, lr=0.001, **opt_kwargs):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        self.optimiser = optimiser_class(params, lr=lr, **opt_kwargs)
        self.loss_fn = loss
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def _set_training(self, mode: bool):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = mode
    
    def fit(self, X, y, epochs=10, batch_size=32, validation_split=0.1, verbose=True):
        """Train the model with mini-batch gradient descent."""
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # Validation split
        n = len(X)
        n_val = int(n * validation_split)
        indices = np.random.permutation(n)
        X_val, y_val = X[indices[:n_val]], y[indices[:n_val]]
        X_train, y_train = X[indices[n_val:]], y[indices[n_val:]]
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(1, epochs + 1):
            self._set_training(True)
            
            # Shuffle
            perm = np.random.permutation(len(X_train))
            X_train, y_train = X_train[perm], y_train[perm]
            
            epoch_loss = 0
            n_batches = 0
            
            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                xb = Tensor(X_train[start:end])
                yb = Tensor(y_train[start:end])
                
                # Forward
                pred = self.forward(xb)
                loss = self.loss_fn(pred, yb)
                
                # Backward
                self.optimiser.zero_grad()
                loss.backward()
                
                # Update
                self.optimiser.step()
                
                epoch_loss += loss.data
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            history['train_loss'].append(float(avg_loss))
            
            # Validation
            self._set_training(False)
            val_pred = self.forward(Tensor(X_val))
            val_loss = self.loss_fn(val_pred, Tensor(y_val))
            history['val_loss'].append(float(val_loss.data))
            
            # Accuracy
            train_pred = self.forward(Tensor(X_train))
            if y_train.ndim > 1 and y_train.shape[1] > 1:
                train_acc = (train_pred.data.argmax(axis=1) == y_train.argmax(axis=1)).mean()
                val_acc = (val_pred.data.argmax(axis=1) == y_val.argmax(axis=1)).mean()
            else:
                train_acc = ((train_pred.data >= 0.5).astype(int).flatten() == y_train.flatten()).mean()
                val_acc = ((val_pred.data >= 0.5).astype(int).flatten() == y_val.flatten()).mean()
            
            history['train_acc'].append(float(train_acc))
            history['val_acc'].append(float(val_acc))
            
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | "
                      f"Acc: {train_acc:.4f} | Val Loss: {val_loss.data:.4f} | Val Acc: {val_acc:.4f}")
        
        return history
    
    def predict(self, X):
        self._set_training(False)
        return self.forward(Tensor(np.array(X, dtype=np.float64))).data
