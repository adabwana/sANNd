
import math
import numpy as np

"""
This module contains common activation functions and their corresponding derivatives
used in Machine Learning and Artificial Neural Networks.
Activation functions are responsible for introducing non-linearity into the model.
"""

def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function.
    f(x) = max(0, x)
    Commonly used in hidden layers for neural networks.

    :param x: Input value.
    :return: Activated output.
    """
    return max(0, x)


def relu_derivative(x):
    """
    Derivative of the ReLU activation function.
    f'(x) = 1 if x > 0, otherwise 0.

    :param x: Input value.
    :return: Derivative of ReLU at input.
    """
    return 1 if x > 0 else 0


def sigmoid(x):
    """
    Sigmoid activation function.
    f(x) = 1 / (1 + exp(-x))
    Commonly used for binary classification outputs.

    :param x: Input value.
    :return: Activated output.
    """
    return 1 / (1 + math.exp(-x))

def np_sigmoid(x):
    """
    Sigmoid activation function.
    f(x) = 1 / (1 + exp(-x))
    Commonly used for binary classification outputs.

    :param x: Input value.
    :return: Activated output.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of the Sigmoid activation function.
    f'(x) = f(x) * (1 - f(x))

    :param x: Input value.
    :return: Derivative of Sigmoid at input.
    """
    sigmoid_val = sigmoid(x)
    return sigmoid_val * (1 - sigmoid_val)


def tanh(x):
    """
    Hyperbolic tangent activation function.
    f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    :param x: Input value.
    :return: Activated output.
    """
    return math.tanh(x)


def tanh_derivative(x):
    """
    Derivative of the Hyperbolic tangent activation function.
    f'(x) = 1 - f(x)^2

    :param x: Input value.
    :return: Derivative of tanh at input.
    """
    tanh_val = tanh(x)
    return 1 - tanh_val ** 2


def softmax(x):
    """
    Softmax activation function.
    f(x) = exp(x_i) / sum(exp(x_j)) for all j
    Commonly used in multi-class classification problems.

    :param x: Input vector (list or array).
    :return: Softmax probability distribution.
    """
    e_x = [math.exp(i) for i in x]
    sum_e_x = sum(e_x)
    return [i / sum_e_x for i in e_x]


def softmax_derivative(x):
    """
    Derivative of the Softmax function.
    Softmax derivative is complex due to the output being a probability distribution.
    It requires Jacobian matrix computation for multi-class predictions.

    :param x: Input vector (list or array).
    :return: Jacobian matrix of Softmax derivatives.
    """
    s = softmax(x)
    jacobian_matrix = [[s[i] * (1 - s[j]) if i == j else -s[i] * s[j] for j in range(len(s))] for i in range(len(s))]
    return jacobian_matrix


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function.
    f(x) = x if x > 0, else alpha * x

    :param x: Input value.
    :param alpha: Slope for negative values (default is 0.01).
    :return: Activated output.
    """
    return x if x > 0 else alpha * x


def leaky_relu_derivative(x, alpha=0.01):
    """
    Derivative of Leaky ReLU activation function.
    f'(x) = 1 if x > 0, else alpha

    :param x: Input value.
    :param alpha: Slope for negative values (default is 0.01).
    :return: Derivative of Leaky ReLU at input.
    """
    return 1 if x > 0 else alpha


def elu(x, alpha=1.0):
    """
    Exponential Linear Unit (ELU) activation function.
    f(x) = x if x > 0, else alpha * (exp(x) - 1)

    :param x: Input value.
    :param alpha: Slope for negative values (default is 1.0).
    :return: Activated output.
    """
    return x if x > 0 else alpha * (math.exp(x) - 1)


def elu_derivative(x, alpha=1.0):
    """
    Derivative of Exponential Linear Unit (ELU) activation function.
    f'(x) = 1 if x > 0, else alpha * exp(x)

    :param x: Input value.
    :param alpha: Slope for negative values (default is 1.0).
    :return: Derivative of ELU at input.
    """
    return 1 if x > 0 else alpha * math.exp(x)


def swish(x, beta=1.0):
    """
    Swish activation function.
    f(x) = x * sigmoid(beta * x)

    :param x: Input value.
    :param beta: Parameter to control the steepness of the curve.
    :return: Activated output.
    """
    return x * sigmoid(beta * x)


def swish_derivative(x, beta=1.0):
    """
    Derivative of Swish activation function.
    f'(x) = sigmoid(beta * x) + x * beta * sigmoid(beta * x) * (1 - sigmoid(beta * x))

    :param x: Input value.
    :param beta: Parameter to control the steepness of the curve.
    :return: Derivative of Swish at input.
    """
    sigmoid_val = sigmoid(beta * x)
    return sigmoid_val + x * beta * sigmoid_val * (1 - sigmoid_val)


def hard_sigmoid(x):
    """
    Hard Sigmoid activation function.
    f(x) = min(max(0, (x + 1) / 2), 1)

    :param x: Input value.
    :return: Activated output.
    """
    return max(0, min(1, (x + 1) / 2))


def hard_sigmoid_derivative(x):
    """
    Derivative of Hard Sigmoid activation function.
    f'(x) = 0 if x <= -1 or x >= 1, else 1/2

    :param x: Input value.
    :return: Derivative of Hard Sigmoid at input.
    """
    if x <= -1 or x >= 1:
        return 0
    else:
        return 0.5


def identity(x):
    """
    Identity function (linear activation).
    f(x) = x

    :param x: Input value.
    :return: Output equals the input.
    """
    return x


def identity_derivative(x):
    """
    Derivative of the Identity activation function.
    f'(x) = 1

    :param x: Input value.
    :return: Derivative of Identity at input.
    """
    return 1


def softplus(x):
    """
    Softplus activation function.
    f(x) = log(1 + exp(x))

    :param x: Input value.
    :return: Activated output.
    """
    return math.log(1 + math.exp(x))


def softplus_derivative(x):
    """
    Derivative of Softplus activation function.
    f'(x) = 1 / (1 + exp(-x))

    :param x: Input value.
    :return: Derivative of Softplus at input.
    """
    return 1 / (1 + math.exp(-x))

def scale(x, y):
    return x * y
