"""
Loss Functions
==============
Author: Veronica Pilagov
"""

from .engine import Tensor


class BinaryCrossEntropy:
    """BCE = -(1/N) * sum[ y*log(p) + (1-y)*log(1-p) ]"""
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        eps = 1e-7
        p = y_pred.clip(eps, 1 - eps)
        loss = (y_true * p.log() + (1 - y_true) * (1 - p).log()) * (-1)
        return loss.mean()


class CategoricalCrossEntropy:
    """CCE = -(1/N) * sum[ y * log(p) ] where y is one-hot."""
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        eps = 1e-7
        p = y_pred.clip(eps, 1 - eps)
        loss = (y_true * p.log()).sum(axis=1).mean() * (-1)
        return loss


class MeanSquaredError:
    """MSE = (1/N) * sum[ (y - p)^2 ]"""
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true
        return (diff * diff).mean()
