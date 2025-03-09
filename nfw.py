import random
import math
from mould import *

# Define activation functions
def scale(x, y):
    return x * y

def add(x, y):
    return x + y

def softplus(x):
    return math.log1p(math.exp(min(x, 50)))  # Prevent overflow

# Gradient functions
def compute_gradient(output, target):
    return [(o - t) * 0.01 for o, t in zip(output, target)]  # Simple derivative

def apply_gradient(grad, param, lr):
    return param - lr * grad  # Learning rate-based update

# Initialize Moulds
input_layer = [0.5]
hw = Mould([-random.uniform(1, 5)], func=lambda x: x, train_func=apply_gradient)
hb = Mould([0.0], func=lambda x: x, train_func=apply_gradient)
ow = Mould([-random.uniform(1, 5)], func=lambda x: x, train_func=apply_gradient)
ob = Mould([0.0], func=lambda x: x, train_func=apply_gradient)

target_output = [1.0348316875442132]

for epoch in range(2000):
    # Forward pass
    ha = Mould(hw, input_layer, func=scale, parent=hw)
    
    ha = Mould(hb, ha, func=add, parent=hb)
    ha = Mould(ha, func=softplus, parent=ha)
    ha_output = list(ha)

    final_output = list(Mould(ob, Mould(ow, ha_output, func=scale, parent=ow), func=add, parent=ob))

    # Compute loss and gradients
    loss = sum((o - t) ** 2 for o, t in zip(final_output, target_output)) / len(final_output)
    gradients = compute_gradient(final_output, target_output)

    # Apply gradients
    ha.train(gradients)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Output: {final_output}, Loss: {loss}")

    # Adjust learning rate dynamically
    if epoch % 500 == 0:
        hw.adjust_learning_rate(0.9)  # Decay learning rate over time
