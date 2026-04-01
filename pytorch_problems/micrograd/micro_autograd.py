import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 

class GradObject:
    def __init__(self, value):
        self.value = value
    
    def __add__(self, other):
        return GradObject(self.value + other.value)
    
    def __mul__(self, other):
        return GradObject(self.value * other.value)
    
    def __repr__(self):
        return f"GradObject({self.value})"

#Equation = x1 + x2 + x1X2
x1 = GradObject(2)
x2 = GradObject(3)

a = x1 * x2
y = x1 + x2 + a 

graph = nx.DiGraph()
graph.add_node("x1",layer=0, value=x1.value)
graph.add_node("x2",layer=0,  value=x2.value)
graph.add_node("a", layer=1, value=a.value)
graph.add_node("y",layer=2,  value=y.value)

graph.add_edge("x1", "a")
graph.add_edge("x2", "a")
graph.add_edge("a", "y")
graph.add_edge("x1", "y")
graph.add_edge("x2", "y")

pos = nx.multipartite_layout(graph, subset_key="layer")
labels = {n: f"{n}\n{graph.nodes[n]['value']}" for n in graph.nodes()}
nx.draw(graph, pos, with_labels=True, labels=labels, node_size=1500, node_color='lightblue', font_size=10)
plt.show()