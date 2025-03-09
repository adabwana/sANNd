"""
Tests for the functional RNN implementation.
"""

import sys
import os
from typing import List, Tuple

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from examples.rnn import (
    create_rnn,
    forward,
    backward,
    reset_state,
    RNNState,
    matrix_multiply
)

@pytest.fixture
def rnn_state() -> RNNState:
    """Create a fresh RNN state for each test"""
    return create_rnn(input_size=2, hidden_size=3, output_size=1)

def describe_RNN():
    def describe_initialization():
        def it_initializes_with_correct_dimensions(rnn_state):
            # Check dimensions of weight matrices in recurrent state
            assert len(rnn_state.recurrent_state.Whh) == 3 * 3  # hidden_size * hidden_size
            assert len(rnn_state.recurrent_state.Wxh) == 2 * 3  # input_size * hidden_size
            assert len(rnn_state.output_weights) == 3  # hidden_size * output_size
            
            # Check dimensions of bias vectors
            assert len(rnn_state.recurrent_state.bh) == 3  # hidden_size
            
            # Check hidden state initialization
            assert len(rnn_state.recurrent_state.hidden_state) == 3  # hidden_size
            assert all(h == 0.0 for h in rnn_state.recurrent_state.hidden_state)  # All zeros
    
    def describe_forward_pass():
        def it_produces_output_with_correct_dimensions(rnn_state):
            x = [1.0, -1.0]  # Example input
            output, new_state = forward(rnn_state, x)
            
            assert len(output) == 1  # output_size
            assert len(new_state.recurrent_state.hidden_state) == 3  # hidden_size
        
        def it_maintains_hidden_state_between_calls(rnn_state):
            x1 = [1.0, -1.0]
            x2 = [-1.0, 1.0]
            
            # First forward pass
            _, state1 = forward(rnn_state, x1)
            # Second forward pass
            _, state2 = forward(state1, x2)
            
            # Hidden states should be different due to recurrent connection
            assert not all(
                h1 == h2 
                for h1, h2 in zip(
                    state1.recurrent_state.hidden_state,
                    state2.recurrent_state.hidden_state
                )
            )
        
        def it_resets_hidden_state_correctly(rnn_state):
            x = [1.0, -1.0]
            
            # Run a forward pass
            _, state = forward(rnn_state, x)
            # Reset hidden state
            new_state = reset_state(state)
            
            assert all(h == 0.0 for h in new_state.recurrent_state.hidden_state)
    
    def describe_backward_pass():
        def it_updates_weights_during_training(rnn_state):
            x = [1.0, -1.0]
            
            # Get initial weights
            initial_weights = list(rnn_state.output_weights)
            
            # Forward pass
            output, state = forward(rnn_state, x)
            # Backward pass with some gradients
            gradients = [0.1]  # For output_size=1
            new_state = backward(state, x, gradients)
            
            # Weights should be updated
            assert not all(w1 == w2 for w1, w2 in zip(initial_weights, new_state.output_weights))
    
    def describe_matrix_operations():
        def it_performs_matrix_multiplication_correctly(rnn_state):
            matrix = [1.0, 2.0, 3.0, 4.0]  # 2x2 matrix
            vector = [0.5, -0.5]
            result = matrix_multiply(matrix, vector, 2, 2)
            
            # Manual calculation for verification
            expected = [
                1.0 * 0.5 + 2.0 * (-0.5),  # First row
                3.0 * 0.5 + 4.0 * (-0.5)   # Second row
            ]
            
            assert len(result) == 2
            assert all(abs(r - e) < 1e-6 for r, e in zip(result, expected))
    
    def describe_training():
        def it_reduces_loss_on_simple_task():
            # Create a simple RNN for binary classification
            state = create_rnn(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
            
            # Simple training data
            training_data = [
                ([1, 0], [1]),
                ([0, 1], [1]),
                ([1, 1], [0]),
                ([0, 0], [0])
            ]
            
            # Calculate initial loss
            initial_loss = 0
            state = reset_state(state)
            for x, target in training_data:
                output, state = forward(state, x)
                initial_loss += sum((o - t) ** 2 for o, t in zip(output, target))
            initial_loss /= len(training_data)
            
            # Train for more epochs with a higher learning rate
            for _ in range(500):
                state = reset_state(state)
                for x, target in training_data:
                    # Forward pass
                    output, state = forward(state, x)
                    
                    # Calculate loss
                    loss = sum((o - t) ** 2 for o, t in zip(output, target)) / len(output)
                    
                    # Backward pass
                    gradients = [(o - t) for o, t in zip(output, target)]  # Removed the 0.1 factor
                    state = backward(state, x, gradients)
            
            # Calculate final loss
            final_loss = 0
            state = reset_state(state)
            for x, target in training_data:
                output, state = forward(state, x)
                final_loss += sum((o - t) ** 2 for o, t in zip(output, target))
            final_loss /= len(training_data)
            
            # Loss should decrease
            assert final_loss < initial_loss

if __name__ == "__main__":
    pytest.main([__file__]) 