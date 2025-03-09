from modulator import *
import math

# Utility functions
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return math.tanh(x)

# LSTM Example using Modulator
sequence = [0.5, -0.2, 0.1]
weights = (-0.5, 0.3, 0.8, -0.1)
biases = (0.1, -0.2, 0.05, 0.0)

h_prev, c_prev = 0.0, 0.0
h_states = []
c_states = []

for x in sequence:
    i = sigmoid(x * weights[0] + biases[0])
    f = sigmoid(x * weights[1] + biases[1])
    o = sigmoid(x * weights[2] + biases[2])
    c_tilde = tanh(x * weights[3] + biases[3])
    c = f * c_prev + i * c_tilde
    h = o * tanh(c)
    h_states.append(h)
    c_states.append(c)
    h_prev, c_prev = h, c

outputs = Modulator(h_states)
last_cell = Modulator(c_states)

print(list(outputs))
print(list(last_cell))

