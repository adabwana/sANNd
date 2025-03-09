"""
Mould: A functional implementation of trainable, modulating iterators.
"""

from typing import Iterable, Callable, Any, Optional, List, TypeVar
from itertools import cycle, islice
from dataclasses import dataclass
from functools import partial

T = TypeVar('T')
TransformFunc = Callable[..., Any]
TrainFunc = Callable[[Any, Any, float], Any]

@dataclass(frozen=True)
class MouldState:
    """Immutable state container for Mould parameters"""
    iterables: List[List[Any]]
    func: TransformFunc
    train_func: TrainFunc
    length: int
    parent: Optional['MouldState']
    learning_rate: float
    momentum: float
    batch_size: int
    gradient_clip: Optional[float]
    velocity: List[float]

def create_mould(
    *iterables: Iterable,
    func: Optional[TransformFunc] = None,
    train_func: Optional[TrainFunc] = None,
    length: Optional[int] = None,
    parent: Optional[MouldState] = None,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    batch_size: int = 1,
    gradient_clip: Optional[float] = None
) -> MouldState:
    """Create a new Mould state"""
    iterables_list = [list(iterable) for iterable in iterables]
    actual_length = length or max(len(it) for it in iterables_list)
    
    return MouldState(
        iterables=iterables_list,
        func=func or (lambda *x: x),
        train_func=train_func or (lambda grad, item, lr: item - lr * grad),
        length=actual_length,
        parent=parent,
        learning_rate=learning_rate,
        momentum=momentum,
        batch_size=batch_size,
        gradient_clip=gradient_clip,
        velocity=[0.0 for _ in range(len(iterables_list))]
    )

def forward(state: MouldState) -> Iterable[Any]:
    """Pure function for forward pass through the Mould"""
    cycled_iterables = [cycle(it) for it in state.iterables]
    sliced_iterables = (islice(it, state.length) for it in cycled_iterables)
    zipped = zip(*sliced_iterables)
    return (state.func(*items) for items in zipped)

def clip_gradients(gradients: List[Any], clip_value: float) -> List[Any]:
    """Pure function to clip gradient values"""
    return [max(min(g, clip_value), -clip_value) for g in gradients]

def batch_gradients(gradients: List[Any], batch_size: int) -> List[Any]:
    """Pure function to compute batched gradients"""
    num_batches = max(1, len(gradients) // batch_size)
    return [sum(gradients[i::num_batches]) / num_batches for i in range(num_batches)]

def update_iterables(
    state: MouldState,
    gradients: List[Any],
    new_velocity: List[float]
) -> List[List[Any]]:
    """Pure function to update iterables with gradients"""
    new_iterables = []
    for iterable in state.iterables:
        new_values = []
        # Use modulo to handle case of differing lengths.
        for j, item in enumerate(iterable):
            grad = gradients[j % len(gradients)]
            new_values.append(state.train_func(grad, item, state.learning_rate))
        new_iterables.append(new_values)
    return new_iterables

def train(state: MouldState, gradients: List[Any]) -> MouldState:
    """Pure function for backward pass through the Mould"""
    # Backpropagate through parent first (if applicable)
    if state.parent:
        parent_state = train(state.parent, gradients)
    else:
        parent_state = None

    # Process gradients
    batched_grads = batch_gradients(gradients, state.batch_size)
    if state.gradient_clip is not None:
        batched_grads = clip_gradients(batched_grads, state.gradient_clip)
    
    # Update velocity with momentum
    new_velocity = [
        state.momentum * v - state.learning_rate * g
        for v, g in zip(state.velocity, batched_grads)
    ]
    
    # Update iterables
    new_iterables = update_iterables(state, batched_grads, new_velocity)
    
    return MouldState(
        iterables=new_iterables,
        func=state.func,
        train_func=state.train_func,
        length=state.length,
        parent=parent_state if state.parent else None,
        learning_rate=state.learning_rate,
        momentum=state.momentum,
        batch_size=state.batch_size,
        gradient_clip=state.gradient_clip,
        velocity=new_velocity
    )

def adjust_learning_rate(state: MouldState, factor: float) -> MouldState:
    """Pure function to adjust the learning rate"""
    return MouldState(
        iterables=state.iterables,
        func=state.func,
        train_func=state.train_func,
        length=state.length,
        parent=state.parent,
        learning_rate=state.learning_rate * factor,
        momentum=state.momentum,
        batch_size=state.batch_size,
        gradient_clip=state.gradient_clip,
        velocity=state.velocity
    )

# For backward compatibility, create a class that wraps the functional interface.
class Mould:
    """Class wrapper around the functional Mould implementation for trainable, composable iterators.

    The constructor automatically evaluates any input layers that are themselves Mould instances,
    letting you build networks in a simple, legoblock–like fashion without managing intermediate results.
    """
    def __init__(self, *iterables, func: Optional[TransformFunc] = None,
                 train_func: Optional[TrainFunc] = None, length: Optional[int] = None,
                 parent: Optional[MouldState] = None, learning_rate: float = 0.01,
                 momentum: float = 0.9, batch_size: int = 1, gradient_clip: Optional[float] = None):
        processed_iterables = []
        for iterable in iterables:
            # If the input is a Mould instance, compute its forward pass immediately.
            if isinstance(iterable, Mould):
                processed_iterables.append(list(iterable))
            else:
                try:
                    # If it is already a list, use it directly.
                    if isinstance(iterable, list):
                        processed_iterables.append(iterable)
                    else:
                        processed_iterables.append(list(iterable))
                except TypeError:
                    processed_iterables.append([iterable])
        self.state = create_mould(
            *processed_iterables,
            func=func,
            train_func=train_func,
            length=length,
            parent=parent,
            learning_rate=learning_rate,
            momentum=momentum,
            batch_size=batch_size,
            gradient_clip=gradient_clip
        )
    
    def __iter__(self):
        return forward(self.state)
    
    def train(self, gradients: List[Any]):
        self.state = train(self.state, gradients)
        return self
    
    def adjust_learning_rate(self, factor: float):
        self.state = adjust_learning_rate(self.state, factor)
        return self
