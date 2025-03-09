import sys
sys.path.append('src/mould/mould')
from mould import create_mould, forward
import math

# Define our callables
def scale(x, y):
    print("scale called with:", x, y)
    return x * y

def add(x, y):
    print("add called with:", x, y)
    return x + y

def softplus(x):
    print("softplus called with:", x)
    return math.log1p(math.exp(x))

# Helper to evaluate a mould's forward pass into a list
def eval_mould(state):
    return list(forward(state))

def pipe(value, *funcs):
    """
    Pipes 'value' through each function in funcs,
    returning the final result.
    """
    for fn in funcs:
        value = fn(value)
    return value


if __name__ == "__main__":
    # Starting input layer of our network
    input_vals = [0.5]
    
    final_output = pipe(
        input_vals,
        # weights connecting input to hidden layer (hw)
        lambda x: list(forward(create_mould([-34.4, -2.52], x, func=scale))),
        # biases for hidden layer nodes (hb)
        lambda x: list(forward(create_mould([2.14, 1.29], x, func=add))),
        # activation for hidden layer (ha) using softplus
        lambda x: list(forward(create_mould(x, func=softplus))),
        # weights to the output layer (ow)
        lambda x: list(forward(create_mould([-1.30, 2.28], x, func=scale))),
        # an aggregation of the output (oc), here simply passing the summed value
        lambda x: list(forward(create_mould([sum(x)], func=lambda x: x))),
        # output bias (ob)
        lambda x: list(forward(create_mould([-0.58], x, func=add)))
    )
    
    print("Final output:", final_output) 