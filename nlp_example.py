import math
from funcs import *  # (Optional, if you have any NLP‐related functions; not needed here)

# ----------------------------
# Core Framework (Base class)
# ----------------------------
class Base:
    def __init__(self, parent=None, name="Base", base=None, connections=None, **definition):
        # Initialize optionals.
        base = base or {}
        self.connections = connections or []
        
        # Merge user definitions into the instance dictionary.
        self.apply(self.__dict__, base if type(base) is dict else base.__dict__)
        self.apply(self.__dict__, definition.get("base", {}))
        self.apply(self.__dict__, definition)

        # Set up identity.
        self.parent = parent or None
        self.name = (f"{parent.get('name')}." if parent else "") + f"{name or definition.get('name') or base.get('name')}" or "Base"
        
        if "term" not in self.__dict__:  # Provide a default term.
            self.term = lambda self, input: input

    def __call__(self, input):
       # Shortcut for external input.
       return self.influence(input)

    def apply(self, target_dict, definition):
        # Reference-aware dict update.
        if type(target_dict) is not dict or type(definition) is not dict:
            return  # Fail-fast

        # Update keys with non-dict values.
        target_dict.update({k: v for k, v in definition.items() if not isinstance(v, dict)})
        
        # Recursively update keys that are dicts.
        for k, v in definition.items():
            if isinstance(v, dict):
                if k in target_dict and isinstance(target_dict[k], dict):
                    self.apply(target_dict[k], v)
                else:
                    target_dict[k] = v

    def connect(self, node_class=None, base={}, **definition):
        # Connect self to another node.
        if not node_class or type(node_class) is dict:
            if type(node_class) is dict:
                base = node_class
            node_class = self.__class__

        if not isinstance(node_class, Base):
            target = node_class(parent=self, base=base, **definition)
        else:
            target = node_class.copy(**definition)

        self.connections.append(target)
        return target

    def copy(self, **definition):
        # Create a copy of the instance dictionary, modified by supplied definitions.
        copy = self.__dict__.copy()
        self.apply(copy, definition)
        return self.__class__(**copy)

    def get(self, index):
        # Expose internal dict's get.
        return self.__dict__.get(index)

    def influence(self, input):
        # Process input and propagate through connections.
        if self.parent:
            output = self.term(self, input)
        else:
            output = input
        
        if self.connections:
            output = sum([conn.influence(output) for conn in self.connections])
        
        if not self.parent:
            output = self.term(self, output)
        
        return output

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
    "term": lambda self, input: "Another day in paradise!" if input == "How are you this morning?" else "I don't understand."
}

# Connect the i/o node to the NLP responder.
nIO.connect(Base, nNLP)

# ----------------------------
# Test the NLP Network
# ----------------------------
test_input = "How are you this morning?"
output = nIO(test_input)
print(f"Input: {test_input}\nOutput: {output}")
