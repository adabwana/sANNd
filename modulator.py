from typing import Iterable, Callable, Any, Optional, List, Tuple
from itertools import islice, zip_longest


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



"""
from typing import Iterable, Callable, Any, Optional
from itertools import cycle, islice, zip_longest

class Modulator:
    def __init__(
        self,
        *iterables: Iterable,
        func: Optional[Callable[..., Any]] = None,
        length: Optional[int] = None
    ):
        self.iterables = iterables
        self.func = func or (lambda x: x)
        self.length = length

    def __iter__(self):
        zipped = zip_longest(*self.iterables)
        if self.length is not None:
            zipped = islice(zipped, self.length)
        return (self.func(*items) for items in zipped)
"""


"""
class Modulator:
    def __init__(
        self,
        *iterables: Iterable,
        func: Optional[Callable[..., Any]] = None
    ):
        self.iterables = [iter(it) for it in iterables]
        self.func = func or (lambda x: x)

    def __iter__(self):
        return (self.func(*items) for items in zip(*self.iterables))
"""

"""
class Modulator:
    def __init__(
        self,
        *iterables: Iterable,
        func: Optional[Callable[..., Any]] = None
    ):
        self.iterables = iterables
        self.func = func or (lambda x: x)

    def __iter__(self):
        return (self.func(*items) for items in zip(*self.iterables))
"""

"""class Modulator:
    def __init__(
        self,
        iterable: Iterable,
        func: Optional[Callable[[Any, Any], Any]] = None,
        previous: Optional["Modulator"] = None
    ):
        self.iterable = iterable
        self.func = func #or (lambda prev, x: x)
        self.previous = previous

    def __iter__(self):
        if self.previous:
            return map(self.func, self.previous, self.iterable)
            #return (self.func(p, x) for p, x in zip(self.previous, self.iterable))
        if self.func:
            return map(self.func, self.iterable)
        return iter(self.iterable)
"""