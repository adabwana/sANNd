"""
Mould: A functional implementation of trainable, modulating iterators.
"""

from .activations import sigmoid, tanh, relu, tanh_derivative
from .mould import (
    create_mould,
    forward,
    train,
    adjust_learning_rate,
    MouldState
)
from .layers import (
    create_layer_state,
    create_recurrent_state,
    forward_layer,
    forward_recurrent,
    LayerState,
    RecurrentState
)

__all__ = [
    'create_mould',
    'forward',
    'train',
    'adjust_learning_rate',
    'MouldState',
    'create_layer_state',
    'create_recurrent_state',
    'forward_layer',
    'forward_recurrent',
    'LayerState',
    'RecurrentState',
    'sigmoid',
    'tanh',
    'relu',
    'tanh_derivative'
] 