from sANNd import *
"""
    Reproduce simple network example demonstrated in StatQuest with Josh Starmer:
        https://www.youtube.com/watch?v=CqOfi41LfDw
"""

output = Base(name="output", values={"weight":[-1.30, 2.28], "bias": [-0.58]})
hidden = Base(name="hidden", values={"weight":[-34.4, -2.52], "bias": [2.14, 1.29]}, connections=[output])
model = Base(name="model", connections=[hidden])

print("output",model(0.5))

'''

aggregate()

u = Unit(scale, weight=[-34.4, -2.52], bias=[2.14, 1.29])
term = partial(scale, input = )
[print(inp * val) for inp, val in zip(cycle(inputs), values)]
'''


class otest:
    def __init__(self, iterable, map_func):
        self.iterable = iterable
        self.map_func = map_func
    def __iter__(self):
        return self.map()
    def map(self):
        return map(self.map_func, self.iterable)