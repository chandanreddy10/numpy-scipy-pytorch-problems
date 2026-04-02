import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

#Program to generate any type of computational graph for the given operations
NODE_SIZE=25000
class node:
    def __init__(self, value, input_nodes=None, op=""):
        self.value = value
        self.grad = 0
        self._input_nodes = input_nodes
        self.op = op

    def __add__(self, other):
        out = node(self.value + other.value, input_nodes=[self, other], op="+")
        return out
    
    def __sub__(self, other):
        out = node(self.value - other.value, input_nodes=[self, other], op="-")
        return out
    
    def __mul__(self, other):
        out = node(self.value * other.value, input_nodes=[self, other], op="*")
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

    def build_graph(self):
        """Build the graph with nodes, edges and levels for easy plotting.
        Recursion traversal for easy plotting."""
        node_edges = []

        def inner_function(n, level):
            if isinstance(n, node) and n._input_nodes is not None:
                for value in n._input_nodes:
                    if isinstance(value, node):
                        inner_function(value, level - 1)
                        if (
                            not {
                                "value": value,
                                "child": value._input_nodes,
                                "level": level,
                            }
                            in node_edges
                        ):
                            node_edges.append(
                                {
                                    "value": value,
                                    "child": value._input_nodes,
                                    "level": level,
                                }
                            )
        inner_function(self, 100)
        node_edges.append({"value": self, "child": self._input_nodes, "level":101 })
        return node_edges

x1 = node(4)
x2 = node(2)
x3 = node(5)

y = x3 - x2 + x1 * x2 + x1 / x2 - x1**2 + x2 ** 2
vertices = y.build_graph()
graph = nx.DiGraph()

for vertex in vertices:
    child = vertex.get("child")
    value = vertex.get("value")
    layer = vertex.get("level")
    graph.add_node(value, layer=layer, value=value.op)
    print(vertex, layer)

    if child is not None:
        for child_node in child:
            graph.add_edge(child_node, value)


pos = nx.multipartite_layout(graph, subset_key="layer")
labels = {n: f"{n}\n{graph.nodes[n]['value']}" for n in graph.nodes()}
plt.figure(figsize=(8, 8))

nx.draw(graph, pos, with_labels=True, labels=labels,
        node_size=NODE_SIZE / len(vertices), node_color='lightblue', font_size=5)

plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)

plt.gca().set_aspect('equal')  
plt.axis('off')    
plt.savefig("misc/graph.jpg", dpi=300)
plt.show()
