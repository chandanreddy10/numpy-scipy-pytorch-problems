import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


class DropoutVisualizer:
    def __init__(self, layers, dropout_rates, total_steps, output_dir, filename):
        self.layers = layers
        self.dropout_rates = dropout_rates
        self.total_steps = total_steps
        self.output_dir = output_dir
        self.filename = filename
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.node_positions = self._compute_node_positions()
        self.nodes = self._build_nodes()
        self.edges = self._build_edges()
        self.step_data = self._simulate_training()

    def _compute_node_positions(self):
        positions = {}
        max_nodes = max(self.layers)
        x_gap = 1.0 / (len(self.layers) - 1)
        for layer_index, node_count in enumerate(self.layers):
            x = layer_index * x_gap
            y_gap = 1.0 / (node_count + 1)
            for node_index in range(node_count):
                y = 1.0 - (node_index + 1) * y_gap
                positions[(layer_index, node_index)] = (x, y)
        return positions

    def _build_nodes(self):
        return list(self.node_positions.keys())

    def _build_edges(self):
        edges = []
        for layer_index in range(len(self.layers) - 1):
            for src_index in range(self.layers[layer_index]):
                for dst_index in range(self.layers[layer_index + 1]):
                    edges.append(((layer_index, src_index), (layer_index + 1, dst_index)))
        return edges

    def _simulate_training(self):
        random.seed(42)
        np.random.seed(42)
        step_data = []
        for step in range(self.total_steps):
            rate = self.dropout_rates[step % len(self.dropout_rates)]
            active_nodes = set(self.nodes)
            dropout_nodes = set()
            for node in self.nodes:
                if random.random() < rate:
                    dropout_nodes.add(node)
            active_nodes -= dropout_nodes
            step_data.append({
                "step": step,
                "dropout_rate": rate,
                "dropout_nodes": dropout_nodes,
                "active_nodes": active_nodes,
            })
        return step_data

    def _draw_graph(self):
        self.ax.clear()
        self.ax.set_title("Neural Network Dropout Training Visualization")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(0, 1.05)

        for edge in self.edges:
            src, dst = edge
            x1, y1 = self.node_positions[src]
            x2, y2 = self.node_positions[dst]
            self.ax.plot([x1, x2], [y1, y2], color="#999999", linewidth=0.8)

        node_x = [self.node_positions[node][0] for node in self.nodes]
        node_y = [self.node_positions[node][1] for node in self.nodes]
        self.node_scatter = self.ax.scatter(node_x, node_y, s=200, facecolors="#ffffff", edgecolors="#333333", linewidths=1.2)

        for layer_index, node_count in enumerate(self.layers):
            self.ax.text(layer_index / (len(self.layers) - 1), .95, f"Layer {layer_index + 1}", ha="center", va="bottom", fontsize=10)
            

    def _update(self, frame):
        data = self.step_data[frame]
        dropout_nodes = data["dropout_nodes"]
        active_nodes = data["active_nodes"]
        colors = []
        sizes = []
        for node in self.nodes:
            if node in dropout_nodes:
                colors.append("#ff6666")
                sizes.append(450)
            elif node in active_nodes:
                colors.append("#66b3ff")
                sizes.append(200)
            else:
                colors.append("#ffffff")
                sizes.append(200)

        self.node_scatter.set_facecolors(colors)
        self.node_scatter.set_edgecolors(["#333333"] * len(self.nodes))
        self.node_scatter.set_sizes(sizes)
        self.ax.set_title(
            f"Step {data['step'] + 1}/{self.total_steps}  Dropout rate: {data['dropout_rate']:.2f}"
        )

    def run(self):
        self._draw_graph()
        anim = FuncAnimation(self.fig, self._update, frames=len(self.step_data), interval=250, repeat=False)

        output_path = os.path.join(self.output_dir, self.filename)
        writer = FFMpegWriter(fps=4)
        anim.save(output_path, writer=writer)
        plt.close(self.fig)
