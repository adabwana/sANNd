"""
Trainable Modulator Module
-------------------------
This module provides a trainable iterator wrapper designed for use in machine learning
pipelines. It applies a user-defined transformation function during iteration and allows
gradient-based updates via a training function.
"""

from typing import Iterable, Callable, Any, Optional, List
from itertools import islice, zip_longest, chain


class Modulator:
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
        iterables = [list(iterable) if isinstance(iterable, Iterable) and not isinstance(iterable, (str, bytes)) else [iterable] for iterable in self.iterables]
        zipped = zip_longest(*iterables)
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
    import math

    def scale(x, y):
        return x * y

    def add(x, y):
        return x + y

    def softplus(x):
        return math.log1p(math.exp(x))

    input_layer = [0.5, 0.5]

    hw = Modulator([-34.4, -2.52], input_layer, func=scale)
    hb = Modulator([2.14, 1.29], hw, func=add)
    ha = Modulator(hb, func=softplus)

    ha_output = list(ha)
    ow = Modulator([-1.30, 2.28], ha_output, func=scale)
    ow_output = list(ow)
    oc = Modulator([sum(ow_output)], func=lambda x: x)
    ob = Modulator([-0.58], oc, func=add)

    print(list(ob))
    # output: [1.0348316875442132]

"""
    # Additional test case
    print("Additional test case:")
    input_layer2 = [1.0, 1.0]
    hw2 = Modulator([-30.0, -3.0], input_layer2, func=scale)
    hb2 = Modulator([3.0, 2.0], hw2, func=add)
    ha2 = Modulator(hb2, func=softplus)
    print(list(ha2))
"""