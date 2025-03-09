from typing import Iterable, Callable, Any, Optional, Dict, List, Tuple
from functools import reduce
import itertools
from collections import defaultdict, deque

class IterableWrapper:
    """A flexible, chainable, and lazy iterable processing pipeline."""

    def __init__(self, iterable: Iterable):
        self.iterable = iterable

    def map(self, func: Callable[[Any], Any]) -> "IterableWrapper":
        """Applies a function to each item in the iterable."""
        return IterableWrapper(map(func, self.iterable))

    def filter(self, func: Callable[[Any], bool]) -> "IterableWrapper":
        """Filters items based on a predicate function."""
        return IterableWrapper(filter(func, self.iterable))

    def reduce(self, func: Callable[[Any, Any], Any], initializer: Optional[Any] = None) -> Any:
        """Reduces the iterable using a function."""
        if initializer is not None:
            return reduce(func, self.iterable, initializer)
        return reduce(func, self.iterable)

    def take(self, n: int) -> "IterableWrapper":
        """Takes the first `n` items from the iterable."""
        return IterableWrapper(itertools.islice(self.iterable, n))

    def chain(self, *others: Iterable) -> "IterableWrapper":
        """Chains multiple iterables together."""
        return IterableWrapper(itertools.chain(self.iterable, *others))

    def batch(self, n: int) -> "IterableWrapper":
        """Yields items in batches of `n`."""
        def batched(iterable):
            it = iter(iterable)
            while (chunk := list(itertools.islice(it, n))):
                yield chunk
        return IterableWrapper(batched(self.iterable))

    def window(self, n: int) -> "IterableWrapper":
        """Yields a sliding window of size `n`."""
        def windows(iterable):
            it = iter(iterable)
            d = deque(itertools.islice(it, n), maxlen=n)
            if len(d) == n:
                yield tuple(d)
            for item in it:
                d.append(item)
                yield tuple(d)
        return IterableWrapper(windows(self.iterable))

    def flatten(self) -> "IterableWrapper":
        """Flattens one level of nested iterables."""
        def flat(iterable):
            for item in iterable:
                yield from item
        return IterableWrapper(flat(self.iterable))

    def unique(self) -> "IterableWrapper":
        """Yields unique items while preserving order."""
        seen = set()
        def uniq(iterable):
            for item in iterable:
                if item not in seen:
                    seen.add(item)
                    yield item
        return IterableWrapper(uniq(self.iterable))

    def group_by(self, key_func: Callable[[Any], Any]) -> Dict[Any, List[Any]]:
        """Groups elements into a dictionary by a key function."""
        groups = defaultdict(list)
        for item in self.iterable:
            groups[key_func(item)].append(item)
        return dict(groups)

    def peek(self, n: int) -> "IterableWrapper":
        """Peeks at the first `n` items without consuming the iterable."""
        a, b = itertools.tee(self.iterable)
        print(list(itertools.islice(a, n)))
        return IterableWrapper(b)

    def compose(self, *funcs: Callable[[Any], Any]) -> "IterableWrapper":
        """Applies multiple functions in sequence."""
        def composed(x):
            for f in funcs:
                x = f(x)
            return x
        return self.map(composed)

    def collect(self) -> List[Any]:
        """Collects the iterable into a list (terminal operation)."""
        return list(self.iterable)

    def __iter__(self):
        return iter(self.iterable)


"""


1️⃣ Basic Chaining
wrapped = (
    IterableWrapper(range(10))
    .filter(lambda x: x % 2 == 0)  # Keep even numbers
    .map(lambda x: x * 3)  # Multiply by 3
    .take(3)  # Get first 3 results
)

print(wrapped.collect())  # Output: [0, 6, 12]

2️⃣ Grouping Data
data = ["apple", "banana", "cherry", "apricot", "blueberry"]

grouped = IterableWrapper(data).group_by(lambda x: x[0])  # Group by first letter
print(grouped)
# Output: {'a': ['apple', 'apricot'], 'b': ['banana', 'blueberry'], 'c': ['cherry']}

3️⃣ Peeking at Values
wrapped = IterableWrapper(range(100)).peek(5)
# Output: [0, 1, 2, 3, 4]

4️⃣ Sliding Window
wrapped = IterableWrapper(range(6)).window(3)
print(wrapped.collect())
# Output: [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)]

5️⃣ Batched Processing
wrapped = IterableWrapper(range(10)).batch(3)
print(wrapped.collect())
# Output: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

6️⃣ Flattening a Nested Iterable
nested = IterableWrapper([[1, 2], [3, 4], [5]])
flattened = nested.flatten()
print(flattened.collect())  
# Output: [1, 2, 3, 4, 5]

"""