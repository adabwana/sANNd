"""
TrainableModulator Module
-------------------------
This module provides a trainable iterator wrapper designed for use in machine learning
pipelines. It applies a user-defined transformation function during iteration and allows
gradient-based updates via a training function.
"""

from typing import Iterable, Callable, Any, Optional, List, Tuple
from itertools import islice, zip_longest


class TrainableModulator:
    def __init__(
        self,
        *iterables: Iterable,
        func: Optional[Callable[..., Any]] = None,
        train_func: Optional[Callable[[Any, Any], Any]] = None,
        length: Optional[int] = None
    ):
        """
        :param iterables: Input iterables to process.
        :param func: Transformation function applied to each set of zipped items.
        :param train_func: Function to apply gradients and update the data.
        :param length: Optional limit on the number of items to process.
        """
        self.iterables = [list(iterable) for iterable in iterables]
        self.func = func or (lambda *x: x)
        self.train_func = train_func or (lambda grad, item: item)
        self.length = length
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        zipped = zip_longest(*self.iterables)
        if self.length is not None:
            zipped = islice(zipped, self.length)
        return (self.func(*items) for items in zipped)

    def train(self, gradients: List[Any]):
        """
        Applies gradients to update the underlying data in-place.

        :param gradients: List of gradients matching the length of each iterable.
        """
        for idx, iterable in enumerate(self.iterables):
            for i, (grad, item) in enumerate(zip(gradients, iterable)):
                iterable[i] = self.train_func(grad, item)


# Example usage
if __name__ == "__main__":
    data = [1.0, 2.0, 3.0]
    labels = [0.0, 1.0, 0.0]

    def forward(x, y):
        return x * 2, y

    def apply_gradient(grad, x):
        return x - grad

    modulator = TrainableModulator(
        data, labels,
        func=forward,
        train_func=apply_gradient
    )

    for epoch in range(30):
        print(f"Epoch {epoch + 1}")
        for output in modulator:
            print(output)
        modulator.train([0.1, 0.1, 0.1])
