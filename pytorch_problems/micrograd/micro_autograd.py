import numpy as np 
import networkx as glib

class GradObject:
    def __init__(self, value):
        self.value = value
    
    def __add__(self, other):
        print("Operator overloading for gradient calculation")
        return GradObject(self.value + other.value)
    def __mul__(self, other):
        return GradObject(self.value * other.value)
    def __repr__(self):
        return f"GradObject({self.value})"

#Example 
x1 = GradObject(2)
x2 = GradObject(3)

print(x1+x2, x1*x2, (x1*x2 + x2*x1))
# y = x1*x2 + x1**2 + np.log(x2)

