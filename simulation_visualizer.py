import math
import numpy as np
import pickle
import copy
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from scipy.spatial import KDTree
from simulation_assembly import MolecularDynamicsSimulation
import argparse

class SimulationAnimator:
    def __init__(self, sim_instance, scaling=1, plot_outer_layer=True, loop=False, last_frame_only=False):
        self.sim_instance = sim_instance
        self.plot_outer_layer = plot_outer_layer
        self.scaling = scaling
        self.loop = loop
        self.last_frame_only = last_frame_only
        self.state_trajectory = sim_instance.state_trajectory
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1, 1, 1])
        self.initialize_animation()

    def initialize_animation(self):
        if self.scaling == 1:
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            self.ax.set_zlabel('z')
        else:
            self.ax.set_xlabel('X (nm)')
            self.ax.set_ylabel('Y (nm)')
            self.ax.set_zlabel('Z (nm)')
        self.ax.set_title('Capsid Assembly Simulation')

        self.scatter = self.ax.scatter([], [], [], s=20, label='Particles')
        max_edges = max(len(state['topology'].edges()) for state in self.state_trajectory)
        self.edge_lines = []
        for _ in range(max_edges * (1 if self.plot_outer_layer else 3)):
            line, = self.ax.plot([], [], [], color='black', alpha=0.3)  # Set alpha for transparency
            self.edge_lines.append(line)
        
        self.anim_running = True
        self.current_frame = 0
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        if self.last_frame_only:
            self.current_frame = len(self.state_trajectory) - 1
            self.update(self.current_frame)
            plt.show()
        else:
            self.timer = self.fig.canvas.new_timer(interval=20)
            self.timer.add_callback(self._update_animation)
            self.animation = FuncAnimation(self.fig, self.update, frames=len(self.state_trajectory), 
                                           interval=5, blit=False, repeat=self.loop, event_source=self.timer)
            self.timer.start()
            plt.show()

    def on_key_press(self, event):
        if self.last_frame_only:
            return
        if event.key == ' ':
            self.toggle_animation()
        elif event.key == 'left':
            self.current_frame = max(0, self.current_frame - 1)
            self.update(self.current_frame)
        elif event.key == 'right':
            self.current_frame = min(len(self.state_trajectory) - 1, self.current_frame + 1)
            self.update(self.current_frame)

    def toggle_animation(self):
        if self.anim_running:
            self.timer.stop()
        else:
            self.timer.start()
        self.anim_running = not self.anim_running

    def _update_animation(self):
        self.current_frame = (self.current_frame + 1) % len(self.state_trajectory)
        self.update(self.current_frame)

    def update(self, frame):
        self.current_frame = frame
        state = self.state_trajectory[frame]
        positions = state['positions']
        topology = state['topology']
        
        node_ids = sorted(topology.nodes())
        node_id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        if self.plot_outer_layer:
            vectors = positions[:, 0, :]
        else:
            vectors = positions.reshape(-1, 3)
        
        xs = vectors[:, 0]*self.scaling
        ys = vectors[:, 1]*self.scaling
        zs = vectors[:, 2]*self.scaling
        
        node_degrees = dict(topology.degree())
        
        # Coloring:
        # 5-fold nodes: red with full opacity
        # 6-fold nodes: gray with alpha=0.1
        # all other degrees: blue with full opacity
        colors = []
        if self.plot_outer_layer:
            for node_id in node_ids:
                degree = node_degrees.get(node_id, 0)
                if degree == 5:
                    colors.append((1, 0, 0, 1))      # red
                elif degree == 6:
                    colors.append((0.5, 0.5, 0.5, 0.1))  # gray with alpha
                else:
                    colors.append((0, 0, 1, 1))      # blue
        else:
            for node_id in node_ids:
                degree = node_degrees.get(node_id, 0)
                if degree == 5:
                    c = (1, 0, 0, 1)
                elif degree == 6:
                    c = (0.5, 0.5, 0.5, 0.1)
                else:
                    c = (0, 0, 1, 1)
                # Repeat for each sub-layer
                colors.extend([c, c, c])
        
        self.scatter._offsets3d = (xs, ys, zs)
        self.scatter.set_color(colors)
        
        for line in self.edge_lines:
            line.set_data([], [])
            line.set_3d_properties([])
        
        current_edges = list(topology.edges())
        edge_count = 0
        for edge in current_edges:
            node1 = edge[0]
            node2 = edge[1]
            idx1 = node_id_map[node1]
            idx2 = node_id_map[node2]
            if self.plot_outer_layer:
                layers = [0]
            else:
                layers = [0, 1, 2]
            for layer in layers:
                pos1 = positions[idx1, layer, :]*self.scaling
                pos2 = positions[idx2, layer, :]*self.scaling
                line = self.edge_lines[edge_count]
                line.set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
                line.set_3d_properties([pos1[2], pos2[2]])
                edge_count += 1
        
        for i in range(edge_count, len(self.edge_lines)):
            line = self.edge_lines[i]
            line.set_data([], [])
            line.set_3d_properties([])
        
        margin = 0
        x_min, x_max = min(xs) - margin, max(xs) + margin
        y_min, y_max = min(ys) - margin, max(ys) + margin
        z_min, z_max = min(zs) - margin, max(zs) + margin
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        self.ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
        self.ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
        self.ax.set_zlim(z_center - max_range / 2, z_center + max_range / 2)
        plt.draw()
        return self.scatter, *self.edge_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animate a molecular dynamics simulation.")
    parser.add_argument('filepath', type=str, help="Path to the simulation state file.")
    parser.add_argument('--last-frame', action='store_true', help="Show only the last frame.")
    args = parser.parse_args()

    filepath = Path(args.filepath).resolve()
    sim_instance = MolecularDynamicsSimulation.load_state(filepath)
    sim_instance.state_trajectory = sim_instance.state_trajectory[-50:]
    scaling_factor = sim_instance.monomer_info['scaling']

    animator = SimulationAnimator(sim_instance, scaling=scaling_factor, plot_outer_layer=True, loop=False, last_frame_only=args.last_frame)

    last_state = sim_instance.state_trajectory[-1]
    topology = last_state['topology']
    degree_array = np.zeros(8, dtype=int)
    for node in topology.nodes():
        degree = topology.degree(node)
        if degree < len(degree_array):
            degree_array[degree] += 1
        else:
            degree_array = np.resize(degree_array, degree + 1)
            degree_array[degree] = 1

    for i in range(len(degree_array)):
        print(f'DEGREE {i}: {degree_array[i]} NODES')
