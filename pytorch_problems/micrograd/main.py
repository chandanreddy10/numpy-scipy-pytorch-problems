import numpy as np 

class GradObject:
    def __init__(self, value):
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
    
    def __mul__(self, other):
        out = GradObject(self.value * other.value)
        out.parents = [self, other]

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        
        out._backward=_backward
        return out

    def backward(self, grad=None):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.parents:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

a = GradObject(np.array(2.0))
b = GradObject(np.array(3.0))

c = a * b + a * b
c.backward()
print(a.grad)
print(b.grad) 

