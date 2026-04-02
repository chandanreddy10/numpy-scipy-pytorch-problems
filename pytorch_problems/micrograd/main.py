import numpy as np 

class GradObject:
    def __init__(self,value):
        self.value = value 
        self.grad = 0
        self.parents = []
        self._backward = lambda : None
    def __add__(self, other):
        out = GradObject(self.value + other.value)
        out.parents = [self, other]

        def _backward():
            self.grad += out.grad
            other.grad += out.grad 
        out._backward = _backward

        return out 
    
    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.value)
        self.grad += grad
        self._backward()

a = GradObject(np.array(2.0))
b = GradObject(np.array(3.0))

c = a + b
c.backward()

print(a.grad)
print(b.grad) 

