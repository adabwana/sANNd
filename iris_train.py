import math
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sANNd import *


# ----------------------------
# Define the Network
# ----------------------------

# Input node: simply passes input onward.
nIO = Base(
    name="i/o node",
    bias=0.0,
    term=lambda self, input: input + self.bias
)

# Connection node: scales its input by a learnable weight.
nCon = {
    "name": "connection",
    "input_term": lambda self, input: input * self.weight
}

# Hidden node: applies a sigmoid activation with bias.
nNeu = {
    "name": "hidden",
    "bias": 0.1,
    "activate": lambda x: 1 / (1 + math.exp(-x)),
    "input_term": lambda self, input: self.activate(input + self.bias)
}

# Output node: simply passes the input (with its own bias, if desired).
nOut = {
    "name": "output",
    "bias": 0.0,
    "input_term": lambda self, input: input + self.bias
}

# Wire up the network:
# nIO -> nCon1 -> nNeu -> nCon2 -> nOut
nIO.connect(Base, nCon, name="nCon1", weight=0.5) \
    .connect(Base, nNeu, name="nNeu", bias=0.1) \
    .connect(Base, nCon, name="nCon2", weight=1.0) \
    .connect(Base, nOut, name="nOut")

# ----------------------------
# Load Public Dataset (Iris)
# ----------------------------
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Use only classes 0 and 1 for binary classification.
indices = (y == 0) | (y == 1)
X = X[indices]
y = y[indices]

# For simplicity, use only the first feature.
X = X[:, 0].reshape(-1, 1)

# Normalize the feature.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Training Setup
# ----------------------------
def mse_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)

learning_rate = 0.01
epochs = 100

# For demonstration, we’ll update the weight in the first connection node only.
# In a full implementation, you would backpropagate through all nodes.
first_con_node = nIO.connections[0]  # This is the node corresponding to "nCon1"

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(epochs):
    epoch_loss = 0.0
    for i in range(len(X_train)):
        # For simplicity, use the first feature value as input.
        input_val = X_train[i, 0]
        target = y_train[i]
        
        # Forward pass: get network output.
        output = nIO(input_val)
        
        # Compute loss.
        loss = mse_loss(target, output)
        epoch_loss += loss
        
        # Compute a rough gradient of the loss with respect to the weight of nCon1.
        grad_loss = mse_loss_derivative(target, output)
        # (This gradient computation is a placeholder. A real implementation would require
        # backpropagation through the entire network.)
        grad_w = grad_loss * input_val  # rudimentary estimate
        
        # Update the weight.
        first_con_node.weight -= learning_rate * grad_w

    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(X_train)}")

# ----------------------------
# Test the Network
# ----------------------------
test_loss = 0.0
for i in range(len(X_test)):
    input_val = X_test[i, 0]
    target = y_test[i]
    output = nIO(input_val)
    loss = mse_loss(target, output)
    test_loss += loss
print(f"\nTest Loss: {test_loss/len(X_test)}")
