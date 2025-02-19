After cloning or downloading pynn.py and funcs.py, import both in your script:

from funcs import *
from pynn import *

Then define the nodes with dicts, or constructor parameters, terms must accept two parameters:

dQS_definition = {
    "name": "test",
    "input_term": lambda self, input: input + len(self.name)
}

nQS = Base(dQS_definition)

or


nQS = Base(name="test", term=lambda self, input: input + len(self.name))

You can now give the node a number and it will perform the term and output the result:

print(nQS.get("name"),nQS(123))
#Output: test 127


Connect to other nodes for a complete network:

nQS.connect(Base, dQS_definition, name="test2", input_term=lambda self, input: input - len(self.name))

print(nQS.connections[0].get("name"),nQS(123))
#Output: test.test2 117

