"""
A functional implementation of a simple RNN using Mould iterators and a clean pipeline.

In this version we use a Clojure-like threading macro (pipe) to chain the 
forward pass, loss/gradient computation, and backward update, allowing us to 
avoid manually managing intermediate state variables.
"""

import sys
import os
from typing import List, Tuple, NamedTuple, Any
from functools import reduce

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mould.activations import sigmoid, tanh_derivative
from mould.mould import (
    create_mould,
    forward as mould_forward,
    train as mould_train,
    MouldState
)
from mould.layers import (
    create_recurrent_state,
    RecurrentState,
    Weights,
    matrix_multiply,
    create_mould_layer
)

class RNNState(NamedTuple):
    """Immutable state container for the RNN"""
    recurrent_state: RecurrentState
    output_mould: MouldState
    learning_rate: float

    @property
    def output_weights(self) -> List[float]:
        """For backward compatibility with tests"""
        return list(self.output_mould.iterables[0])

def create_rnn(input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.01) -> RNNState:
    """Create a new RNN state with a Mould-based output layer"""
    return RNNState(
        recurrent_state=create_recurrent_state(input_size, hidden_size),
        output_mould=create_mould_layer(hidden_size, output_size),
        learning_rate=learning_rate
    )

def forward(state: RNNState, x: Weights) -> Tuple[Weights, RNNState]:
    """Pure function for forward pass through the RNN"""
    # Forward through recurrent layer
    Whh_h = matrix_multiply(
        state.recurrent_state.Whh,
        state.recurrent_state.hidden_state,
        state.recurrent_state.hidden_size,
        state.recurrent_state.hidden_size
    )
    
    Wxh_x = matrix_multiply(
        state.recurrent_state.Wxh,
        x,
        state.recurrent_state.hidden_size,
        len(x)
    )
    
    # Combine the recurrent contribution with the input and bias,
    # then apply the sigmoid activation.
    hidden = [
        sigmoid(h + x_val + b)
        for h, x_val, b in zip(Whh_h, Wxh_x, state.recurrent_state.bh)
    ]
    
    # Forward through output layer using Mould: we inject the current hidden layer.
    output_state = MouldState(
        iterables=[state.output_mould.iterables[0], [hidden]],
        func=state.output_mould.func,
        train_func=state.output_mould.train_func,
        length=1,
        parent=state.output_mould.parent,
        learning_rate=state.output_mould.learning_rate,
        momentum=state.output_mould.momentum,
        batch_size=state.output_mould.batch_size,
        gradient_clip=state.output_mould.gradient_clip,
        velocity=state.output_mould.velocity
    )
    output = list(mould_forward(output_state))[0]
    
    # Update recurrent state with the new hidden layer
    new_recurrent_state = state.recurrent_state._replace(hidden_state=hidden)
    new_state = state._replace(recurrent_state=new_recurrent_state)
    
    # Return the output and the updated state
    return output, new_state

def backward(state: RNNState, x: Weights, gradients: Weights) -> RNNState:
    """Pure function for backward pass through the RNN"""
    # Train the output layer using Mould
    new_output_mould = mould_train(state.output_mould, gradients)
    
    # Compute gradients for the hidden layer based on the output weights.
    hidden_state = state.recurrent_state.hidden_state
    hidden_grads = []
    for i in range(state.recurrent_state.hidden_size):
        grad = 0.0
        for j, g in enumerate(gradients):
            grad += g * state.output_weights[i]
        hidden_grads.append(grad)
    
    # Apply the derivative of the sigmoid activation: h * (1 - h)
    hidden_grads = [
        g * h * (1 - h)
        for g, h in zip(hidden_grads, hidden_state)
    ]
    
    # Scale gradients by the learning rate.
    hidden_grads = [g * state.learning_rate for g in hidden_grads]
    
    # Compute gradients for the recurrent weights (Whh)
    Whh_grads = []
    for i in range(state.recurrent_state.hidden_size):
        for j in range(state.recurrent_state.hidden_size):
            grad = hidden_grads[i] * hidden_state[j]
            Whh_grads.append(grad)
    
    # Compute gradients for the input weights (Wxh)
    Wxh_grads = []
    for i in range(state.recurrent_state.hidden_size):
        for j in range(len(x)):
            grad = hidden_grads[i] * x[j]
            Wxh_grads.append(grad)
    
    # Update recurrent weights and biases.
    new_Whh = [w - g for w, g in zip(state.recurrent_state.Whh, Whh_grads)]
    new_Wxh = [w - g for w, g in zip(state.recurrent_state.Wxh, Wxh_grads)]
    new_bh  = [b - g for b, g in zip(state.recurrent_state.bh, hidden_grads)]
    
    new_recurrent_state = state.recurrent_state._replace(
        Whh=new_Whh,
        Wxh=new_Wxh,
        bh=new_bh
    )
    
    return state._replace(
        recurrent_state=new_recurrent_state,
        output_mould=new_output_mould
    )

def reset_state(state: RNNState) -> RNNState:
    """Pure function to reset the RNN's hidden state"""
    new_recurrent = state.recurrent_state._replace(
        hidden_state=[0.0] * state.recurrent_state.hidden_size
    )
    return state._replace(recurrent_state=new_recurrent)

def pipe(value, *funcs):
    """Apply a series of functions to a value."""
    return reduce(lambda x, f: f(x), funcs, value)

def compute_loss(output: List[float], target: List[float]) -> float:
    """Compute mean squared loss between the output and the target."""
    return sum((o - t) ** 2 for o, t in zip(output, target)) / len(output)

def compute_gradients(output: List[float], target: List[float]) -> List[float]:
    """Compute a simple gradient as (output-target) scaled by 0.1."""
    return [(o - t) * 0.1 for o, t in zip(output, target)]

if __name__ == "__main__":
    # Initialize RNN state
    rnn_state = create_rnn(input_size=2, hidden_size=4, output_size=1)
    
    # Training data: XOR-like temporal problem
    training_data = [
        ([1, 0], [1]),
        ([0, 1], [1]),
        ([1, 1], [0]),
        ([0, 0], [0])
    ]
    
    for epoch in range(1000):
        total_loss = 0
        rnn_state = reset_state(rnn_state)
        
        for x, target in training_data:
            # Use the pipe to thread the state through the forward pass,
            # then compute the loss and gradients, and finally perform 
            # the backward update.
            rnn_state, loss = pipe(
                rnn_state,
                lambda state: forward(state, x),  # returns a tuple: (output, updated_state)
                lambda t: (t[1], t[0], compute_loss(t[0], target), compute_gradients(t[0], target)),
                # Now we have a tuple: (state, output, loss, gradients)
                lambda tup: (backward(tup[0], x, tup[3]), tup[2])
            )
            total_loss += loss
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(training_data)}") 