import random
import math


def dot(x, y):
    """Compute the dot product of two vectors."""
    return sum(a * b for a, b in zip(x, y))

def mat_vec_mul(W, x):
    """Multiply matrix W (list of lists) with vector x."""
    result = []
    for row in W:
        s = 0.0
        for w, xi in zip(row, x):
            s += w * xi
        result.append(s)
    return result

def vector_add(x, y):
    """Elementwise addition of two vectors."""
    return [xi + yi for xi, yi in zip(x, y)]

def scalar_vector_mul(scalar, x):
    """Multiply a vector x by a scalar."""
    return [scalar * xi for xi in x]

def relu(x):
    """ReLU activation."""
    return [max(0, xi) for xi in x]

def softmax(x):
    """Compute softmax on a vector."""
    exps = [math.exp(xi) for xi in x]
    sum_exps = sum(exps)
    return [xi / sum_exps for xi in exps]

def softplus(x):
    return math.log(1 + math.exp(x))

def d_softplus(x): # derivative of softplus
    return 1 / (1 + math.exp(-x))

def scale(x, y):
    return x * y

