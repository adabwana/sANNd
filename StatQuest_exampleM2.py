from mould import *
import math

"""
    Reproduce simple network example demonstrated in StatQuest with Josh Starmer:
        https://www.youtube.com/watch?v=CqOfi41LfDw
"""

# Utility functions
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

# Example network
#input = Mould([0.5])
input = [0.5, 0.5]
#print(list(input))
hw = Mould([-34.4, -2.52], input, func=scale)
#print(list(hw))
hb = Mould([2.14, 1.29], hw, func=add)
#print("hb",list(hb))
ha = Mould(hb, func=softplus)

ow = Mould([-1.30, 2.28], ha, func=scale)
oc = Mould([sum(ow)], func=lambda x: x)
ob = Mould([-0.58], oc, func=add)

print(list(ob))


"""
def scale(x,y):
    return x * y

def add(x,y):
    return x + y

def softplus(x):
    \"""
    Softplus activation function.
    f(x) = log(1 + exp(x))

    :param x: Input value.
    :return: Activated output.
    \"""
    return math.log(1 + math.exp(x))


def softplus_derivative(x):
    \"""
    Derivative of Softplus activation function.
    f'(x) = 1 / (1 + exp(-x))

    :param x: Input value.
    :return: Derivative of Softplus at input.
    \"""
    return 1 / (1 + math.exp(-x))


input = Modulator([0.5])

hw = Modulator([-34.4, -2.52], input, func=scale)
hb = Modulator([2.14, 1.29], hw, func=add)
ha = Modulator(hb, func=softplus)

ow = Modulator([-1.30, 2.28], ha, func=scale)
oc = Modulator(ow, func=sum)
ob = Modulator([-0.58], oc, func=add)

print(list(ob))
"""

"""
input = Modulator([0.5])
print(list(input))

hw = Modulator([-34.4, -2.52], scale, input)
print(list(hw))
hb = Modulator([2.14, 1.29], add, hw)
print(list(hb))
ha = Modulator(hb, softplus)
print(list(ha))


ow = Modulator([-1.30, 2.28], scale, ha)
print(list(ow))
#oc = Modulator(ow, sum)
oc = sum(ow)
print(list(oc))
ob = Modulator([-0.58], add, oc)

print(list(ob))
"""