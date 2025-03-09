import math
from mould import Mould  # Import core Mould class

# 🎯 Define Activation and Transformation Functions
def scale(x, y): 
    return x * y  # Multiply weight * input

def add(x, y): 
    return x + y  # Add bias

def softplus(x): 
    return math.log1p(math.exp(min(x, 50)))  # Prevent overflow

# 🎯 Input Value
input_value = [0.5]  # Single input for the network

# 🎯 Shared Node Processing Function
def process_node(weight, bias, scale_factor):
    scaled = Mould(weight, input_value, func=scale, parent=weight)  # Scale input
    added = Mould(bias, scaled, func=add, parent=bias)  # Add bias
    activated = Mould(added, func=softplus, parent=added)  # Softplus activation
    return Mould(activated, scale_factor, func=scale)  # Scale output

# 🎯 Process Both Nodes
node1 = process_node(Mould([-34.4]), Mould([2.14]), Mould([-1.30]))
node2 = process_node(Mould([-2.52]), Mould([1.29]), Mould([2.28]))

# 🎯 Summation Node (Final Output)
summed = Mould(node1, node2, func=add)  # Sum node outputs
final_output = Mould(Mould([-0.58]), summed, func=add)  # Add final bias

# 🎯 Compute and Print Output
output_result = list(final_output)[0]
print(f"\n🎯 Final Output: {output_result}, Expected: 1.0348316875442132")
