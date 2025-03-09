import numpy as np
from sANNd import Base

# Define sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# ----------------------------
# FeedForward Layer using sANNd Base
# ----------------------------
class FFLayer(Base):
    def __init__(self, input_size, output_size, learning_rate=0.1, name="ff layer"):
        # Initialize weights and biases
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.zeros((output_size, 1))
        self.learning_rate = learning_rate
        
        # Set activation functions (using sigmoid for this example)
        self.activation = sigmoid
        self.activation_deriv = sigmoid_deriv
        
        # Placeholders for caching during forward pass (for backprop)
        self.last_input = None
        self.last_z = None
        
        # Initialize the Base node with a forward term
        super().__init__(name=name, term=self.forward)
    
    def forward(self, input):
        """
        Compute the layer’s output given an input column vector.
        """
        self.last_input = input  # Cache the input for backpropagation
        z = np.dot(self.weights, input) + self.biases
        self.last_z = z        # Cache the linear combination
        a = self.activation(z)
        return a
    
    def backward(self, grad_output):
        """
        Backpropagate the error through this layer.
        
        grad_output: The gradient of the loss with respect to the layer's output.
        
        Returns the gradient with respect to the layer's input.
        """
        # Compute gradient w.r.t. z using chain rule:
        grad_z = grad_output * self.activation_deriv(self.last_z)
        
        # Gradients for weights and biases
        grad_weights = np.dot(grad_z, self.last_input.T)
        grad_biases = grad_z  # For each output neuron
        
        # Compute gradient with respect to the input for further backpropagation
        grad_input = np.dot(self.weights.T, grad_z)
        
        # Update parameters
        self.weights -= self.learning_rate * grad_weights
        self.biases  -= self.learning_rate * grad_biases
        
        return grad_input

# ----------------------------
# FeedForward Network Definition
# ----------------------------
class FeedForwardNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Create two layers: one hidden and one output.
        self.hidden = FFLayer(input_size, hidden_size, learning_rate, name="hidden layer")
        self.output = FFLayer(hidden_size, output_size, learning_rate, name="output layer")
    
    def forward(self, x):
        """
        Perform a forward pass through the network.
        """
        a1 = self.hidden.forward(x)
        a2 = self.output.forward(a1)
        return a2
    
    def train(self, x, y):
        """
        Train the network on one sample (x, y).
        Uses Mean Squared Error (MSE) as the loss.
        """
        # Forward pass
        output = self.forward(x)
        
        # Compute MSE loss and its gradient w.r.t. output
        loss = np.mean((output - y) ** 2)
        grad_loss = 2 * (output - y) / y.size
        
        # Backpropagation: first through output layer, then hidden
        grad_hidden = self.output.backward(grad_loss)
        self.hidden.backward(grad_hidden)
        
        return loss

# ----------------------------
# Example Training on XOR Problem
# ----------------------------
def main():
    # XOR inputs and outputs
    # Each column is one sample.
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])  # Shape: (2, 4)
    Y = np.array([[0, 1, 1, 0]])  # Shape: (1, 4)
    
    # Initialize network: 2 inputs, 3 hidden neurons, 1 output
    ffnet = FeedForwardNetwork(input_size=2, hidden_size=3, output_size=1, learning_rate=0.1)
    
    epochs = 10000
    for epoch in range(epochs):
        epoch_loss = 0
        # Iterate over each training sample (stochastic gradient descent)
        for i in range(X.shape[1]):
            x_sample = X[:, i].reshape(-1, 1)
            y_sample = Y[:, i].reshape(-1, 1)
            loss = ffnet.train(x_sample, y_sample)
            epoch_loss += loss
        # Print average loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss / X.shape[1]:.4f}")
    
    # Test the trained network
    print("\nTesting trained network on XOR inputs:")
    for i in range(X.shape[1]):
        x_sample = X[:, i].reshape(-1, 1)
        output = ffnet.forward(x_sample)
        print(f"Input: {X[:, i]}  Predicted Output: {output.flatten()[0]:.4f}")

if __name__ == "__main__":
    main()
