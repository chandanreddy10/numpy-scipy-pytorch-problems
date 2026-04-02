import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 
import random 

#Expand the Project to general purporse computational graph generator
class GradObject:
    def __init__(self, value, name, input_nodes=None):
        self.value = value
        self.grad = 0
        self._input_nodes = input_nodes
        self.name = name

    def __add__(self, other):
        name = random.choice(["a","b","c","d","e"])
        return GradObject(self.value + other.value, name, input_nodes=[self.name, other.name])
    
    def __mul__(self, other):
        name = random.choice(["a","b","c","d","e"])
        return GradObject(self.value * other.value, name, input_nodes=[self.name, other.name])
    
    def __repr__(self):
        return f"GradObject({self.value})"

#Equation = x1 + x2 + x1X2
x1 = GradObject(2, "x1")
x2 = GradObject(3, "x2")

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