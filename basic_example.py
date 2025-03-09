import random
import math
from mould import Mould

# 🎯 Define Activation and Transformation Functions
def scale(x, y): 
    return x * y  # Multiply weight * input

def add(x, y): 
    return x + y  # Add bias

def softplus(x): 
    return math.log1p(math.exp(min(x, 50)))  # Prevent overflow

# 🎯 Define Gradient Computation & Training Functions
def compute_gradient(output, target): 
    return [(o - t) * 0.01 for o, t in zip(output, target)]  # Basic derivative

def apply_gradient(grad, param, lr): 
    return param - lr * grad  # Gradient descent update

# 🎯 Initialize Network Components
input_layer = [0.5]  # Input data
target_output = [1.0348]  # Target output

# 🔗 Initialize Moulds (Weights & Biases)
hw = Mould([-random.uniform(1, 5)], func=lambda x: x, train_func=apply_gradient)  # Hidden layer weight
hb = Mould([0.0], func=lambda x: x, train_func=apply_gradient)  # Hidden layer bias
ow = Mould([-random.uniform(1, 5)], func=lambda x: x, train_func=apply_gradient)  # Output weight
ob = Mould([0.0], func=lambda x: x, train_func=apply_gradient)  # Output bias

# 🎯 Train the Model
for epoch in range(2000):
    # 🏗 Forward Pass
    ha = Mould(hw, input_layer, func=scale, parent=hw)  # Weighted input
    ha = Mould(hb, ha, func=add, parent=hb)  # Add bias
    ha = Mould(ha, func=softplus, parent=ha)  # Activation function
    ha_output = list(ha)  # Convert iterator to list

    final_output = list(Mould(ob, Mould(ow, ha_output, func=scale, parent=ow), func=add, parent=ob))

    # 🏗 Compute Loss and Gradients
    loss = sum((o - t) ** 2 for o, t in zip(final_output, target_output)) / len(final_output)
    gradients = compute_gradient(final_output, target_output)

    # 🔄 Backpropagation (Train Moulds)
    ha.train(gradients)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Output: {final_output}, Loss: {loss}")

# 🎯 Final Prediction
print(f"\n🎉 Final Output: {final_output}, Target: {target_output}")
