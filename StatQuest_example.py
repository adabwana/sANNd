from pynn import *
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
