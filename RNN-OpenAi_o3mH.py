#! Suggested by ChatGPT 03-mini-high

# Input/Output node (as in your feed-forward example).
nIO = Base(
    name="i/o node",
    bias=-0.58,
    term=lambda self, input: input + self.bias
)

# Define a recurrent cell as a dictionary.
# The cell carries its own "state" and uses a custom term to update:
#   new_state = activate( input + (recurrence_weight * previous_state) + bias )
# (The lambda uses a helper inline lambda to update self.state and return the new state.)
nRNN = {
    "name": "rnn cell",
    "state": 0.0,               # initial state
    "recurrence_weight": 0.8,   # weight on previous state
    "bias": 0.1,                # bias for the recurrent cell
    "activate": math.tanh,      # activation function (could also use softplus, etc.)
    "term": lambda self, input: 
         (lambda prev, new: (setattr(self, 'state', new), new)[1])(
             self.state, 
             self.activate(input + self.recurrence_weight * self.state + self.bias)
         )
}

# Connect the i/o node to the recurrent cell.
rnn_node = nIO.connect(Base, nRNN)

# Add a simple feedback connection to the rnn cell.
# This connection simply passes its input through unchanged, effectively feeding the cell’s own output back.
rnn_node.connect(Base, {"name": "feedback", "term": lambda self, input: input})

# -------------------------------
# Simulate a Sequence (Time Steps)
# -------------------------------

# In a recurrent network, repeated calls update the internal state.
sequence = [0.5, 0.2, 0.8, 0.1, 0.9]
print("Simulating RNN over sequence:")
for t, inp in enumerate(sequence):
    # Call the network from the i/o node; this cascades the input through the rnn cell.
    output = nIO(inp)
    print(f"Time step {t}: input = {inp:.2f}, output = {output:.4f}, rnn state = {rnn_node.state:.4f}")