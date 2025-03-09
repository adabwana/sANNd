from typing import Iterable, Callable, Any, Optional, List
from itertools import islice, zip_longest
import math

class Mould:
    """
    Mould: A trainable, modulating iterator that shapes data flow in the sANNd framework.

    - Applies transformations during iteration.
    - Supports momentum-based training with adjustable learning rates.
    - Maintains linkage to a parent Mould for backpropagation.
    - Implements batch updates and gradient clipping for stability.
    """

    def __init__(
        self,
        *iterables: Iterable,
        func: Optional[Callable[..., Any]] = None,
        train_func: Optional[Callable[..., Any]] = None,
        length: Optional[int] = None,
        parent: Optional["Mould"] = None,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        batch_size: int = 1,
        gradient_clip: Optional[float] = None
    ):
        self.iterables = [list(iterable) for iterable in iterables]  # Convert iterables to lists
        self.func = func or (lambda *x: x)  # Default: identity function
        self.train_func = train_func or (lambda grad, item, lr: item - lr * grad)  # Default: simple gradient step
        self.length = length  # Optional iteration limit
        self.parent = parent  # Link to previous Mould in the chain
        
        # Training parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip
        
        # Velocity for momentum-based updates
        self.velocity = [0.0 for _ in range(len(self.iterables))]

        self.epoch = 0  # Track training iterations

    def __iter__(self):
        """Forward pass: Apply func to each zipped set of items from the iterables."""
        self.epoch += 1
        zipped = zip_longest(*self.iterables)
        if self.length is not None:
            zipped = islice(zipped, self.length)
        return (self.func(*items) for items in zipped)

    def train(self, gradients: List[Any]):
        """
        Backward pass: Applies gradients using momentum, learning rate, and optional gradient clipping.
        - Aggregates gradients over batch_size before applying updates.
        - Uses parent linkage to backpropagate gradients.
        """
        if self.parent:
            self.parent.train(gradients)  # Backpropagate first
        
        # Process gradients in batches
        num_batches = max(1, len(gradients) // self.batch_size)
        batched_gradients = [sum(gradients[i::num_batches]) / num_batches for i in range(num_batches)]
        
        # Apply gradient clipping if enabled
        if self.gradient_clip:
            batched_gradients = [max(min(g, self.gradient_clip), -self.gradient_clip) for g in batched_gradients]

        # Update iterables using momentum-based training
        for i, iterable in enumerate(self.iterables):
            new_values = []
            for j, item in enumerate(iterable):
                grad = batched_gradients[j] if j < len(batched_gradients) else 0
                
                # Momentum update rule: velocity = momentum * velocity - learning_rate * gradient
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
                updated_value = self.train_func(grad, item, self.learning_rate)
                
                new_values.append(updated_value)
            
            self.iterables[i] = new_values  # Update parameters

    def adjust_learning_rate(self, factor: float):
        """Dynamically adjust the learning rate."""
        self.learning_rate *= factor
