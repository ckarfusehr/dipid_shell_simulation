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

        # Instead of using sim_instance.state_trajectory directly in plotting,
        # we store it here for convenience.
        self.state_trajectory = sim_instance.state_trajectory

        # Create the figure and the 3D axis
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1, 1, 1])

        # Initialize the animation
        self.initialize_animation()

    def initialize_animation(self):
        # Set labels based on the scaling
        if self.scaling == 1:
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            self.ax.set_zlabel('z')
        else:
            self.ax.set_xlabel('X (nm)')
            self.ax.set_ylabel('Y (nm)')
            self.ax.set_zlabel('Z (nm)')

        self.ax.set_title('Capsid Assembly Simulation')

        # Initialize an empty scatter plot
        self.scatter = self.ax.scatter([], [], [], s=20, label='Particles')

        # Determine the maximum possible number of edges in any state
        max_edges = 0
        for state in self.state_trajectory:
            num_edges = len(state['topology'].edges())
            if num_edges > max_edges:
                max_edges = num_edges

        # If we plot only the outer layer, each edge is drawn once;
        # otherwise, we have 3 layers per node, so each edge is drawn 3 times.
        factor = 1 if self.plot_outer_layer else 3

        # Prepare a list of line handles for each possible edge to animate
        self.edge_lines = []
        for _ in range(max_edges * factor):
            line, = self.ax.plot([], [], [], color='black', alpha=0.3)
            self.edge_lines.append(line)

        self.anim_running = True
        self.current_frame = 0

        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # If only the last frame is required, show it and exit
        if self.last_frame_only:
            self.current_frame = len(self.state_trajectory) - 1
            self.update(self.current_frame)
            plt.show()
        else:
            # Otherwise set up a timer-based animation
            self.timer = self.fig.canvas.new_timer(interval=20)
            self.timer.add_callback(self._update_animation)

            # Create the FuncAnimation, but let the timer call 'update'
            self.animation = FuncAnimation(
                self.fig,
                self.update,
                frames=len(self.state_trajectory),
                interval=5,
                blit=False,
                repeat=self.loop,
                event_source=self.timer
            )
            self.timer.start()
            plt.show()

    def on_key_press(self, event):
        # If we are only showing the last frame, do nothing
        if self.last_frame_only:
            return

        # Spacebar toggles play/pause
        if event.key == ' ':
            self.toggle_animation()

        # Left arrow -> go to previous frame
        elif event.key == 'left':
            self.current_frame = max(0, self.current_frame - 1)
            self.update(self.current_frame)

        # Right arrow -> go to next frame
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
        # Increment the frame index and wrap around if looping
        self.current_frame = (self.current_frame + 1) % len(self.state_trajectory)
        self.update(self.current_frame)

    def update(self, frame):
        """
        This is the main function that updates the plot for each frame.
        It reconstructs a 'positions_plot' array that matches exactly
        the nodes in the current state's topology, thus avoiding any
        mismatch in node count versus positions.
        """
        self.current_frame = frame
        state = self.state_trajectory[frame]

        topology = state['topology']
        node_ids = sorted(topology.nodes())
        node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

        # map_node_config can exist in the state dictionary or in self.sim_instance,
        # depending on how the simulation is stored. We will check for it in the state first.
        map_node_config = state.get('map_node_config', None)
        if map_node_config is None:
            # Fallback if it is only stored in sim_instance
            map_node_config = self.sim_instance.map_node_config

        # Original positions array might be larger than the number of nodes if a merge happened.
        # We build a new positions_plot that strictly matches the sorted node_ids in the topology.
        original_positions = state['positions']  # shape: (N, 3, 3) possibly bigger than len(node_ids)

        positions_plot = np.zeros((len(node_ids), 3, 3), dtype=original_positions.dtype)
        for i, node_id in enumerate(node_ids):
            original_index = map_node_config[node_id]
            positions_plot[i, :, :] = original_positions[original_index, :, :]

        # Now positions_plot has shape = (number_of_nodes_in_topology, 3, 3).
        # If we plot only the outer layer, that is layer index 0. Otherwise all layers.
        if self.plot_outer_layer:
            vectors = positions_plot[:, 0, :]  # shape: (num_nodes, 3)
        else:
            vectors = positions_plot.reshape(-1, 3)  # shape: (num_nodes * 3, 3)

        xs = vectors[:, 0] * self.scaling
        ys = vectors[:, 1] * self.scaling
        zs = vectors[:, 2] * self.scaling

        # Color the nodes depending on their degree
        node_degrees = dict(topology.degree())

        colors = []
        if self.plot_outer_layer:
            # We have one color per node
            for node_id in node_ids:
                degree = node_degrees.get(node_id, 0)
                if degree == 5:
                    colors.append((1, 0, 0, 1))           # red
                elif degree == 6:
                    colors.append((0.5, 0.5, 0.5, 0.1))   # gray with alpha=0.1
                else:
                    colors.append((0, 0, 1, 1))           # blue
        else:
            # We have 3 layers per node, so we replicate the color three times
            for node_id in node_ids:
                degree = node_degrees.get(node_id, 0)
                if degree == 5:
                    color_tuple = (1, 0, 0, 1)          # red
                elif degree == 6:
                    color_tuple = (0.5, 0.5, 0.5, 0.1)  # gray with alpha
                else:
                    color_tuple = (0, 0, 1, 1)          # blue
                colors.extend([color_tuple, color_tuple, color_tuple])

        # Update the scatter plot positions and colors
        self.scatter._offsets3d = (xs, ys, zs)
        self.scatter.set_color(colors)

        # Clear all previous line data
        for line in self.edge_lines:
            line.set_data([], [])
            line.set_3d_properties([])

        # Plot edges with the newly constructed positions_plot
        current_edges = list(topology.edges())
        edge_count = 0
        for edge in current_edges:
            node1 = edge[0]
            node2 = edge[1]

            # Map node IDs to their new positions_plot indices
            index1 = node_id_to_index[node1]
            index2 = node_id_to_index[node2]

            # Decide which layers to plot
            if self.plot_outer_layer:
                layers = [0]
            else:
                layers = [0, 1, 2]

            for layer in layers:
                position_1 = positions_plot[index1, layer, :] * self.scaling
                position_2 = positions_plot[index2, layer, :] * self.scaling

                line = self.edge_lines[edge_count]
                line.set_data([position_1[0], position_2[0]],
                              [position_1[1], position_2[1]])
                line.set_3d_properties([position_1[2], position_2[2]])
                edge_count += 1

        # In case the new state has fewer edges than the maximum
        # we clear any leftover lines
        for i in range(edge_count, len(self.edge_lines)):
            line = self.edge_lines[i]
            line.set_data([], [])
            line.set_3d_properties([])

        # Auto-scale the axes so the structure fits in the box
        margin = 0
        x_min, x_max = min(xs) - margin, max(xs) + margin
        y_min, y_max = min(ys) - margin, max(ys) + margin
        z_min, z_max = min(zs) - margin, max(zs) + margin
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        z_center = (z_min + z_max) / 2.0

        self.ax.set_xlim(x_center - max_range / 2.0, x_center + max_range / 2.0)
        self.ax.set_ylim(y_center - max_range / 2.0, y_center + max_range / 2.0)
        self.ax.set_zlim(z_center - max_range / 2.0, z_center + max_range / 2.0)

        plt.draw()
        return self.scatter, *self.edge_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animate a molecular dynamics simulation.")
    parser.add_argument('filepath', type=str, help="Path to the simulation state file.")
    parser.add_argument('--last-frame', action='store_true', help="Show only the last frame.")
    arguments = parser.parse_args()

    filepath = Path(arguments.filepath).resolve()
    simulation_instance = MolecularDynamicsSimulation.load_state(filepath)

    # Attempt to keep the map_node_config and positions in sync
    simulation_instance.cleanup_map_node_config()

    # Only show the last 50 frames
    simulation_instance.state_trajectory = simulation_instance.state_trajectory[-50:]

    # Obtain the scaling factor from the simulation data
    scaling_factor = simulation_instance.monomer_info['scaling']

    # Create and run the animator
    animator = SimulationAnimator(
        simulation_instance,
        scaling=scaling_factor,
        plot_outer_layer=True,
        loop=False,
        last_frame_only=arguments.last_frame
    )

    # Example diagnostic checks
    last_state = simulation_instance.state_trajectory[-1]

    topology = last_state['topology']
    degree_array = np.zeros(8, dtype=int)
    for node in topology.nodes():
        node_degree = topology.degree(node)
        if node_degree < len(degree_array):
            degree_array[node_degree] += 1
        else:
            degree_array = np.resize(degree_array, node_degree + 1)
            degree_array[node_degree] = 1

    for index in range(len(degree_array)):
        print(f"DEGREE {index}: {degree_array[index]} NODES")
