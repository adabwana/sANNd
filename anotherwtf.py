"""
Trainable Modulator Module
-------------------------
This module provides a trainable iterator wrapper designed for use in machine learning
pipelines. It applies a user-defined transformation function during iteration and allows
gradient-based updates via a training function, leveraging the chain rule for effective weight and bias adjustments.
"""

from typing import Iterable, Callable, Any, Optional, List
from itertools import islice, zip_longest
import random
import math


class Modulator:
    def __init__(
        self,
        *iterables: Iterable,
        func: Optional[Callable[..., Any]] = None,
        train_func: Optional[Callable[..., Any]] = None,
        length: Optional[int] = None,
        parent: Optional["Modulator"] = None
    ):
        self.iterables = [list(iterable) for iterable in iterables]
        self.func = func or (lambda *x: x)
        self.train_func = train_func or (lambda grads, items: items)
        self.length = length
        self.parent = parent  # Link to previous Modulator
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        zipped = zip_longest(*self.iterables)
        if self.length is not None:
            zipped = islice(zipped, self.length)
        return (self.func(*items) for items in zipped)

    def train(self, gradients: List[Any]):
        """
        Applies gradients in reverse through the chain using the parent linkage.
        """
        if self.parent:
            self.parent.train(gradients)
        for i, iterable in enumerate(self.iterables):
            self.iterables[i] = [self.train_func(grad, item) for grad, item in zip(gradients, iterable)]


if __name__ == "__main__":

    def scale(x, y):
        return x * y

    def add(x, y):
        return x + y

    def softplus(x):
        return math.log1p(math.exp(x))

    def compute_gradient(output, target):
        return [(o - t) * 0.1 for o, t in zip(output, target)]

    def apply_gradient(grad, param):
        return param - grad

    # Initialize weights randomly and biases to zero
    input_layer = [0.5]
    hw = Modulator([-random.uniform(1, 5)], func=lambda x: x, train_func=apply_gradient)
    hb = Modulator([0.0], func=lambda x: x, train_func=apply_gradient)
    ow = Modulator([-random.uniform(1, 5)], func=lambda x: x, train_func=apply_gradient)
    ob = Modulator([0.0], func=lambda x: x, train_func=apply_gradient)

    target_output = [1.0348316875442132]

    for epoch in range(1000):
        # Forward pass
        ha = Modulator(hw, input_layer, func=scale, parent=hw)
        ha = Modulator(hb, ha, func=add, parent=hb)
        ha = Modulator(ha, func=softplus, parent=ha)
        ha_output = list(ha)
        final_output = list(Modulator(ob, Modulator(ow, ha_output, func=scale, parent=ow), func=add, parent=ob))

        # Compute loss
        gradients = compute_gradient(final_output, target_output)
        
        # Apply gradients following chain linkage
        ha.train(gradients)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Output: {final_output}")

    print(f"Final Output: {final_output}, Target: {target_output}")
