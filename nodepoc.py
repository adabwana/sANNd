import random
import math

from funcs import *  # custom script containing activation functions (softplus, softmax, relu, etc.)

class Base:
    def __init__(self, parent=None, name="Base", base=None, connections=None, **definition):
        # Initialize optionals
        base = base or {}
        self.connections = connections or []
        
        # Store all definitions in the instance dictionary - support user customization.
        self.apply(self.__dict__, base if type(base) is dict else base.__dict__)
        self.apply(self.__dict__, definition.get("base"))
        self.apply(self.__dict__, definition)

        # Setup identity.
        self.parent = parent or None
        self.name = (f"{parent.get('name')}." if parent else "") + f"{name or definition.get('name') or base.get('name')}" or "Base"
        
        if "term" not in self.__dict__: # Provide default term.
            self.term = lambda self, input: input

    def __call__(self, input):
       # Shortcut for influence/intended external input interface.
       return self.influence(input)

    def apply(self, target_dict, definition):
        # Reference-aware dict update.
        if type(target_dict) is not dict or type(definition) is not dict:
            return # Fail-fast

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
        # Connect output/self to another node.
        if not node_class or type(node_class) is dict:
            if type(node_class) is dict:
                base = node_class

            node_class = self.__class__

        # First positional arg can be either a class or instanciated variable descended from Base.
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
        # Expose internal dict's get function.
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

# EXAMPLE USAGE #
"""
    Reproduce simple network example demonstrated in StatQuest with Josh Starmer:
        https://www.youtube.com/watch?v=CqOfi41LfDw
"""

# Create Input/Output node/network interface
nIO = Base(
    name = "i/o node",
    bias = -0.58,
    term = lambda self, input: input + self.bias
)

# Define common dictionaries.
nNeu = {
    "name": "hidden ",
    "activate": softplus,
    "term": lambda self, input: input + self.bias
}

nAct = {
    "name": "softplus",
    "term": lambda self, input: softplus(input)
}

nCon = {
    "name": "connection ",
    "term": lambda self, input: scale(input, self.weight) # -scale is defined in funcs.py as x * y
}

# Connect nodes.
nIO.connect(Base, nCon, name=nCon.get("name") + "1", weight = -34.4) \
   .connect(Base, nNeu, name=nNeu.get("name") + "1", bias = 2.14) \
   .connect(Base, nAct) \
   .connect(Base, nCon, name=nCon.get("name") + "1", weight = -1.30)

nIO.connect(Base, nCon, name=nCon.get("name") + "2", weight = -2.52) \
   .connect(Base, nNeu, name=nNeu.get("name") + "2", bias = 1.29) \
   .connect(Base, nAct) \
   .connect(Base, nCon, name=nCon.get("name") + "1", weight = 2.28) 

# nIO is called, returning the network prediction.
print(f"\nOutput: {nIO(0.5)}\n")
# END EXAMPLE USAGE#