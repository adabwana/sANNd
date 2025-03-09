"""
Neural network layer implementations and utilities.
Functional style implementation focusing on data transformation and composition.
"""

import random
from typing import List, Dict, Any, Optional, Callable, TypeVar, NamedTuple, Tuple

from .activations import sigmoid, tanh, relu
from .mould import create_mould, MouldState

# Type definitions for clarity
T = TypeVar('T')
Weights = List[float]
Activation = Callable[[float], float]

class LayerState(NamedTuple):
    """Immutable state container for layer parameters"""
    weights: Weights
    biases: Weights
    activation: Activation

class RecurrentState(NamedTuple):
    """Immutable state container for recurrent layer parameters"""
    Whh: Weights
    Wxh: Weights
    bh: Weights
    hidden_state: Weights
    hidden_size: int
    input_size: int
    activation: Activation

# Activation function mapping
activation_functions = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu
}

def init_weights(input_size: int, output_size: int) -> Weights:
    """Initialize weights with Xavier/Glorot initialization"""
    scale = (2.0 / (input_size + output_size)) ** 0.5
    return [random.uniform(-scale, scale) for _ in range(input_size * output_size)]

def init_biases(size: int) -> Weights:
    """Initialize biases to zero"""
    return [0.0] * size

def create_layer_state(input_size: int, output_size: int, activation: str = "sigmoid") -> LayerState:
    """Create an immutable layer state with initialized parameters"""
    return LayerState(
        weights=init_weights(input_size, output_size),
        biases=init_biases(output_size),
        activation=activation_functions.get(activation, sigmoid)
    )

def create_recurrent_state(input_size: int, hidden_size: int, activation: str = "tanh") -> RecurrentState:
    """Create an immutable recurrent layer state with initialized parameters"""
    # Initialize weight matrices with correct dimensions
    # Whh: hidden_size x hidden_size matrix for hidden-to-hidden connections
    # Wxh: input_size x hidden_size matrix for input-to-hidden connections
    # bh: hidden_size vector for biases
    return RecurrentState(
        Whh=init_weights(hidden_size, hidden_size),  # hidden_size x hidden_size
        Wxh=init_weights(input_size, hidden_size),   # input_size x hidden_size
        bh=init_biases(hidden_size),                 # hidden_size
        hidden_state=init_biases(hidden_size),       # hidden_size
        hidden_size=hidden_size,
        input_size=input_size,
        activation=activation_functions.get(activation, tanh)
    )

def matrix_multiply(matrix: Weights, vector: Weights, rows: int, cols: int) -> Weights:
    """Pure function for matrix multiplication"""
    if len(vector) != cols:
        raise ValueError(f"Vector size {len(vector)} does not match matrix columns {cols}")
    
    result = []
    for i in range(rows):
        row_sum = 0.0
        for j in range(cols):
            if i * cols + j < len(matrix):
                row_sum += matrix[i * cols + j] * vector[j]
        result.append(row_sum)
    return result

def forward_layer(state: LayerState, inputs: Weights) -> Weights:
    """Pure function for forward pass through a layer"""
    preactivations = [
        sum(w * x for w, x in zip(state.weights[i::len(inputs)], inputs)) + state.biases[i]
        for i in range(len(state.biases))
    ]
    return [state.activation(x) for x in preactivations]

def forward_recurrent(state: RecurrentState, inputs: Weights) -> Tuple[Weights, RecurrentState]:
    """Pure function for forward pass through a recurrent layer"""
    # Calculate hidden state components
    Whh_h = matrix_multiply(state.Whh, state.hidden_state, state.hidden_size, state.hidden_size)
    Wxh_x = matrix_multiply(state.Wxh, inputs, state.hidden_size, len(inputs))
    
    # Combine components and apply activation
    hidden_preact = [
        h + x + b 
        for h, x, b in zip(Whh_h, Wxh_x, state.bh)
    ]
    new_hidden = [state.activation(h) for h in hidden_preact]
    
    # Return new hidden state and updated layer state
    new_state = state._replace(hidden_state=new_hidden)
    return new_hidden, new_state

def create_trainable_layer(input_size: int, output_size: int, activation: str = "sigmoid") -> LayerState:
    """Create a trainable layer state"""
    return create_layer_state(input_size, output_size, activation)

def create_mould_layer(input_size: int, output_size: int, activation: str = "sigmoid") -> MouldState:
    """Create a trainable layer as a Mould iterator"""
    scale = (2.0 / (input_size + output_size)) ** 0.5
    weights = [random.uniform(-scale, scale) for _ in range(input_size * output_size)]
    
    def forward_func(weights: List[float], inputs: List[float]) -> List[float]:
        """Forward pass through the layer"""
        # Unpack inputs if they're in a nested list
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        
        # Handle individual weights vs weight list
        if not isinstance(weights, list):
            weights = [weights]
        
        # Compute matrix multiplication
        raw_outputs = matrix_multiply(weights, inputs, output_size, len(inputs))
        
        # Apply activation
        activation_func = activation_functions.get(activation, sigmoid)
        return [activation_func(out) for out in raw_outputs]
    
    def train_func(grad: float, param: float, lr: float) -> float:
        """Update weights using gradient descent"""
        return param - lr * grad
    
    return create_mould(
        weights,  # First iterable: weights
        func=forward_func,
        train_func=train_func,
        learning_rate=0.01
    ) 