from typing import Callable
from numpy import ndarray

def gradient_descent(
        start : ndarray, 
        learning_rate : float,
        partial_derivatives : list[Callable],
        converged : Callable
    ) -> ndarray:
    w = start
    while not converged(w):
        for i, _ in enumerate(w):
            w[i] = w[i] - learning_rate * partial_derivatives[i](w)
    return w