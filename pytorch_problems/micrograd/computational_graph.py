import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 
import random 

class node:
    def __init__(self, value, input_nodes=None, op=""):
        self.value = value
        self.grad = 0
        self._input_nodes = input_nodes
        self.op = op

    def __add__(self, other):
        out = node(self.value+other.value, input_nodes=[self, other], op="+")
        return out
    
    def __mul__(self, other):
        out = node(self.value*other.value, input_nodes=[self, other], op="*")
        return out
    
    def __truediv__(self, other):
        out = node(self.value / other.value, input_nodes=[self, other], op="/")
        return out 
    
    def __floordiv__(self, other):
        out = node(self.value // other.value, input_nodes=[self, other], op="//")
        return out 
    
    def __pow__(self, other):
        out = node(self.value**other, input_nodes=[self], op="^")
        return out 
    
    def __repr__(self):
        return f"node({self.value})"

#Equation = x1 + x2 + x1X2
x1 = node(2, "x1")
x2 = node(3, "x2")

a = x1 * x2
d = x1 + x2
y = d + a

graph = nx.DiGraph()
graph.add_node("x1",layer=0, value=f"Value:{x1.value}\nGrad:{x1.grad}")
graph.add_node("x2",layer=0,  value=f"Value:{x2.value}\nGrad:{x2.grad}")
graph.add_node(f"{a.name}",layer=1,  value=f"Value:{a.value}\nGrad:{a.grad}")
graph.add_node(f"{d.name}",layer=1,  value=f"Value:{d.value}\nGrad:{d.grad}")
graph.add_node(f"{y.name}",layer=2,  value=f"Value:{y.value}\nGrad:{y.grad}")

for edge in a._input_nodes:
    graph.add_edge(edge, f"{a.name}")
    graph.add_edge(edge, f"{a.name}")

for edge in d._input_nodes:
    graph.add_edge(edge, f"{d.name}")
    graph.add_edge(edge, f"{d.name}")

for edge in y._input_nodes:
    graph.add_edge(edge, f"{y.name}")
    graph.add_edge(edge, f"{y.name}")

pos = nx.multipartite_layout(graph, subset_key="layer")
labels = {n: f"{n}\n{graph.nodes[n]['value']}" for n in graph.nodes()}
nx.draw(graph, pos, with_labels=True, labels=labels, node_size=5000, node_color='lightblue', font_size=10)
# plt.savefig("misc/first_graph.jpg", dpi=300)
plt.show()