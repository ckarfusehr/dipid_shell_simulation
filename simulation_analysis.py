import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Particle:
    def __init__(self, m, x, y, z, dt, T_C):
        self.m = m
        self.T_K = T_C + 273.15
        self.position = np.array([x, y, z], dtype=np.float64)
        self.velocity = self.initVelBoltzmann()
        self.acceleration = np.zeros(3, dtype=np.float64)
        self.position_old = self.position - self.velocity * dt
        
    def getPos(self):
        return self.position

    def initVelBoltzmann(self):
        k_B = 1.38e-23
        sigma = np.sqrt(k_B * self.T_K / self.m)
        vel = np.random.normal(0, sigma, (3,)).astype(np.float64)
        return vel

def load_state(file):
    with open(file, 'rb') as f:
        state_trajectory = pickle.load(f)
    return state_trajectory

def plot(state_trajectory, plot_outer_layer=True, loop=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Capsid Assembly Simulation')

    scatter = ax.scatter([], [], [], s=20, label='Particles')
    edge_lines = []
    max_edges = max(len(state['topological_state'].edges()) for state in state_trajectory)

    for _ in range(max_edges):
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
        elif event.key == 'right':
            current_frame[0] = min(len(state_trajectory) - 1, current_frame[0] + 1)  # Go one frame forward
        update(current_frame[0])  # Manually update the animation to the new frame

    def update(frame):
        current_frame[0] = frame  # Update current frame tracker
        physical_state = state_trajectory[frame]['physical_state']
        topological_state = state_trajectory[frame]['topological_state']
        
        xs, ys, zs, colors = [], [], [], []
        for node_id, layers in physical_state.items():
            layer_indices = [0] if plot_outer_layer else [0, 1, 2]
            for layer_index in layer_indices:
                particle = layers[layer_index]
                pos = particle.getPos()
                xs.append(pos[0])
                ys.append(pos[1])
                zs.append(pos[2])
                degree = topological_state.degree(node_id)
                colors.append('red' if degree == 5 else 'blue' if degree == 6 else 'gray')

        scatter._offsets3d = (xs, ys, zs)
        scatter.set_color(colors)

        current_edges = list(topological_state.edges())
        for line in edge_lines:
            line.set_data([], [])
            line.set_3d_properties([])

        for i, edge in enumerate(current_edges):
            plot_layers = 3 if not plot_outer_layer else 1
            for j in range(plot_layers):
                pos1 = physical_state[edge[0]][j].getPos()
                pos2 = physical_state[edge[1]][j].getPos()
                edge_lines[i].set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
                edge_lines[i].set_3d_properties([pos1[2], pos2[2]])
                edge_lines[i].set_color('black')

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
    animation = FuncAnimation(fig, update, frames=frames, interval=10, blit=False, repeat=loop)
    
    fig.canvas.mpl_connect('key_press_event', toggle_animation)  # Space bar for pausing/resuming
    fig.canvas.mpl_connect('key_press_event', jump_frame)  # Left/Right arrow keys for frame control
    plt.show()
    
    return animation

# Load and animate the trajectory
filename = 'sim_2.pkl'
cd = Path().resolve()
filepath = cd / filename

state_trajectory = load_state(filepath)
anim = plot(state_trajectory, plot_outer_layer=True, loop=False)
