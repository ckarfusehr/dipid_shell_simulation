import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

def load_state(file):
    with open(file, 'rb') as f:
        state_trajectory = pickle.load(f)
    return state_trajectory

def plot(state_trajectory, plot_outer_layer=True, loop=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Capsid Assembly Simulation')

    scatter = ax.scatter([], [], [], s=20, label='Particles')
    edge_lines = []
    max_edges = max(len(state['topology'].edges()) for state in state_trajectory)

    # Create a pool of line objects for edges
    for _ in range(max_edges * (1 if plot_outer_layer else 3)):
        line, = ax.plot([], [], [], color='black')
        edge_lines.append(line)

    anim_running = [True]  # Mutable flag to control animation state
    current_frame = [0]  # Track the current frame index

    def toggle_animation(event):
        if event.key == ' ':
            if anim_running[0]:
                animation.event_source.stop()
            else:
                animation.event_source.start()
            anim_running[0] = not anim_running[0]

    def jump_frame(event):
        if event.key == 'left':
            current_frame[0] = max(0, current_frame[0] - 1)  # Go one frame back, no negative index
            update(current_frame[0])  # Manually update the animation to the new frame
        elif event.key == 'right':
            current_frame[0] = min(len(state_trajectory) - 1, current_frame[0] + 1)  # Go one frame forward
            update(current_frame[0])  # Manually update the animation to the new frame

    def update(frame):
        current_frame[0] = frame  # Update current frame tracker
        positions = state_trajectory[frame]['positions']  # positions is a NumPy array of shape (N_nodes, 3, 3)
        topology = state_trajectory[frame]['topology']  # topology is a NetworkX graph

        # Reconstruct the node_id_map
        node_ids = sorted(topology.nodes())
        node_id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}

        N_nodes = positions.shape[0]
        if plot_outer_layer:
            # Use the top layer positions
            vectors = positions[:, 0, :]  # Shape: (N_nodes, 3)
        else:
            # Use all layers
            vectors = positions.reshape(-1, 3)  # Shape: (N_nodes * 3, 3)

        xs = vectors[:, 0]
        ys = vectors[:, 1]
        zs = vectors[:, 2]

        # Colors based on node degree
        node_degrees = dict(topology.degree())
        colors = []
        if plot_outer_layer:
            for idx, node_id in enumerate(node_ids):
                degree = node_degrees.get(node_id, 0)
                if degree == 5:
                    colors.append('red')
                elif degree == 6:
                    colors.append('blue')
                else:
                    colors.append('gray')
        else:
            for idx, node_id in enumerate(node_ids):
                degree = node_degrees.get(node_id, 0)
                for _ in range(3):  # Repeat color for each layer
                    if degree == 5:
                        colors.append('red')
                    elif degree == 6:
                        colors.append('blue')
                    else:
                        colors.append('gray')

        scatter._offsets3d = (xs, ys, zs)
        scatter.set_color(colors)

        # Clear previous edges
        for line in edge_lines:
            line.set_data([], [])
            line.set_3d_properties([])

        # Update edges
        current_edges = list(topology.edges())
        edge_count = 0
        for edge in current_edges:
            node1 = edge[0]
            node2 = edge[1]
            idx1 = node_id_map[node1]
            idx2 = node_id_map[node2]
            if plot_outer_layer:
                layers = [0]
            else:
                layers = [0, 1, 2]
            for layer in layers:
                pos1 = positions[idx1, layer, :]
                pos2 = positions[idx2, layer, :]
                line = edge_lines[edge_count]
                line.set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
                line.set_3d_properties([pos1[2], pos2[2]])
                edge_count += 1

        # Hide unused edge lines
        for i in range(edge_count, len(edge_lines)):
            line = edge_lines[i]
            line.set_data([], [])
            line.set_3d_properties([])

        # Adjust plot limits
        margin = 0
        x_min, x_max = min(xs) - margin, max(xs) + margin
        y_min, y_max = min(ys) - margin, max(ys) + margin
        z_min, z_max = min(zs) - margin, max(zs) + margin
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
        ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
        ax.set_zlim(z_center - max_range / 2, z_center + max_range / 2)

        return scatter, *edge_lines

    frames = len(state_trajectory)
    animation = FuncAnimation(fig, update, frames=frames, interval=20, blit=False, repeat=loop)
    
    fig.canvas.mpl_connect('key_press_event', toggle_animation)  # Space bar for pausing/resuming
    fig.canvas.mpl_connect('key_press_event', jump_frame)  # Left/Right arrow keys for frame control
    plt.show()
    
    return animation

# Load and animate the trajectory
filename = '20241108145546_sim_langevin_dt0.01_delta0.2_km1_damping1.pkl'
cd = Path().resolve()
filepath = cd / filename

state_trajectory = load_state(filepath)
anim = plot(state_trajectory[-1:], plot_outer_layer=True, loop=False)

topology = state_trajectory[-1]['topology']

degree_array = np.zeros(8)

for node in topology.nodes():
    degree_array[topology.degree(node)] += 1
        
        
for i in range(len(degree_array)):
    print(f'DEGREE {i}: {degree_array[i]} NODES')