from mould import *
import math

"""
    Reproduce simple network example demonstrated in StatQuest with Josh Starmer:
        https://www.youtube.com/watch?v=CqOfi41LfDw
"""

# Callables
def scale(x, y):
    print("scale", x, y)
    return x * y

def add(x, y):
    print("add", x, y)
    return x + y

def softplus(x):
    print("softplus", x)
    return math.log1p(math.exp(x))

def softplus_derivative(x):
    return 1 / (1 + math.exp(-x))

# Bee - beep - boop - bee - boop
input = [0.5] 
hw = Mould([-34.4, -2.52], input, func=scale)
hb = Mould([2.14, 1.29], hw, func=add)
ha = Mould(hb, func=softplus)

ow = Mould([-1.30, 2.28], ha, func=scale)
oc = Mould([sum(ow)], func=lambda x: x)
ob = Mould([-0.58], oc, func=add)

# Bam!
print(list(ob))
