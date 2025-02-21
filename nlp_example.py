import math
from funcs import *  # (Optional, if you have any NLP‐related functions; not needed here)
from sANNd import *

# ----------------------------
# NLP Network Example
# ----------------------------

# Create Input/Output node that simply passes along the input.
nIO = Base(
    name="i/o node",
    term=lambda self, input: input  # Identity for textual input.
)

# Define an NLP responder node as a dictionary.
# If the input text exactly matches "How are you this morning?",
# it returns "Another day in paradise!" otherwise it returns a default reply.
nNLP = {
    "name": "nlp responder",
    "input_term": lambda self, input: "Another day in paradise!" if input == "How are you this morning?" else "I don't understand."
}

# Connect the i/o node to the NLP responder.
nIO.connect(Base, nNLP)

# ----------------------------
# Test the NLP Network
# ----------------------------
test_input = "How are you this morning?"
output = nIO(test_input)
print(f"Input: {test_input}\nOutput: {output}")
