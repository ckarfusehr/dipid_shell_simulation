import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Literal
from simulation_assembly import MolecularDynamicsSimulation
from scipy.optimize import least_squares
import numpy as np
from tqdm import tqdm

class Ellipsoid:
    def __init__(self, positions, topology, scaling=1):
        self.positions = positions * scaling
        self.topology = topology
        self.scaling = scaling

        node_ids = sorted(topology.nodes())
        self.node_id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}

        self.fig = None
        self.ax = None

        self.x = None
        self.y = None
        self.z = None

        self.center = None
        self.radii = None
        self.rotation_matrix = None

    def combine_topology_position(self):
        G = nx.Graph()
        for node_id, pos_idx in self.node_id_map.items():
            G.add_node(node_id, pos=np.array(self.positions[pos_idx, :]))
        G.add_edges_from(self.topology.edges(data=True))
        return G

    def triangle_area(self, p1, p2, p3):
        return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

    def calc_surface_area(self, mode: Literal['triangulation', 'fit'], print=False):
        topology = self.combine_topology_position()

        if mode == 'triangulation':
            surface_area_triangulation = 0
            # Only use cliques of size exactly 3
            for clique in nx.find_cliques(topology):
                if len(clique) == 3:
                    p1, p2, p3 = (topology.nodes[node]['pos'] for node in clique)
                    surface_area_triangulation += self.triangle_area(p1, p2, p3)
            if print:
                print(f'Surface area (Triangulation): {surface_area_triangulation}')
            return surface_area_triangulation

        elif mode == 'fit':
            a, b, c = self.radii
            p = 1.6075
            term = ((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3
            surface_area_fit = 4 * np.pi * (term ** (1 / p))
            if print:
                print(f'Surface Area (Fit): {surface_area_fit}')
            return surface_area_fit

    def fit_ellipsoid_pca(self):

        node_ids = sorted(self.topology.nodes())
        valid_positions = []
        for node_id in node_ids:
            pos_idx = self.node_id_map[node_id]  # This maps the node to its correct row index
            valid_positions.append(self.positions[pos_idx, :])
        valid_positions = np.array(valid_positions)

        self.center = valid_positions.mean(axis=0)
        positions_centered = valid_positions - self.center
        cov_matrix = np.cov(positions_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.rotation_matrix = eigenvectors

        projected_data = positions_centered @ eigenvectors
        min_proj = projected_data.min(axis=0)
        max_proj = projected_data.max(axis=0)
        ranges = max_proj - min_proj
        self.radii = ranges / 2.0
        
    def fit_ellipsoid_pca_custom_diameters(self):
        """
        1) Perform PCA-based orientation of the point cloud.
        2) For each principal axis (x, y, z in the PCA space),
        compute the distance along that axis between the points
        closest to that axis on its negative and positive side.
        """

        # 1. Collect node positions
        node_ids = sorted(self.topology.nodes())
        valid_positions = []
        for node_id in node_ids:
            pos_idx = self.node_id_map[node_id]  # Maps node to its correct row index
            valid_positions.append(self.positions[pos_idx, :])
        valid_positions = np.array(valid_positions)

        # 2. Compute the center of mass and center the data
        self.center = valid_positions.mean(axis=0)
        positions_centered = valid_positions - self.center

        # 3. PCA: compute covariance and eigen-decomposition
        cov_matrix = np.cov(positions_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues/eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store the principal axes (rotation matrix)
        self.rotation_matrix = eigenvectors

        # 4. Project data onto principal axes
        #    Each column in projected_data is the coordinate along one PCA axis:
        #       projected_data[:, 0] -> coordinate along principal axis 1 (largest eigenvalue)
        #       projected_data[:, 1] -> coordinate along principal axis 2
        #       projected_data[:, 2] -> coordinate along principal axis 3 (smallest eigenvalue)
        projected_data = positions_centered @ eigenvectors

        # 5. For convenience, define a helper that measures the extent
        #    specifically along axis i (0->x, 1->y, 2->z in PCA space)
        def axis_extent(data, axis_i):
            """
            Returns the difference in 'axis_i' coordinate between:
            - the negative-side point closest to that axis, and
            - the positive-side point closest to that axis.
            """

            coords = data[:, axis_i]  # coordinate along axis_i
            # Distance from axis_i means Euclidean distance in the other two dimensions
            # For example, if axis_i == 2 (the 'z-axis'), distance to it is sqrt(x^2 + y^2).
            # We'll do it generally by ignoring the axis_i coordinate:
            #   Step 1: extract all coordinates except axis_i
            #   Step 2: compute norm in that 2D subspace for each point.
            #   Step 3: find the point with minimal distance among negative coords, then among positive coords.

            # Indices of the other two dimensions
            dims = [0, 1, 2]
            dims.remove(axis_i)

            # Distances from this principal axis in the orthogonal plane
            dist_to_axis = np.sqrt((data[:, dims] ** 2).sum(axis=1))

            # Negative side = coords < 0, positive side = coords > 0
            neg_mask = coords < 0
            pos_mask = coords > 0

            # If there are no points on negative side or no points on positive side,
            # we cannot define an extent. Handle gracefully:
            if not np.any(neg_mask) or not np.any(pos_mask):
                # Could return 0.0 or np.nan if there's no valid extent in this axis.
                return 0.0

            # Among negative coords, find the point with minimal distance to the axis
            idx_neg = np.argmin(dist_to_axis[neg_mask])
            # Among positive coords, find the point with minimal distance to the axis
            idx_pos = np.argmin(dist_to_axis[pos_mask])

            # We need the actual global indices in the original array
            neg_indices = np.where(neg_mask)[0]
            pos_indices = np.where(pos_mask)[0]

            # The negative-side point closest to the axis
            neg_closest_idx = neg_indices[idx_neg]
            # The positive-side point closest to the axis
            pos_closest_idx = pos_indices[idx_pos]

            # Now get their coordinate along the axis
            coord_neg = coords[neg_closest_idx]
            coord_pos = coords[pos_closest_idx]

            # The extent is the difference in axis_i coordinate
            return coord_pos - coord_neg

        # 6. Compute the extents along each axis (PCA: x, y, z)
        extents = [axis_extent(projected_data, i) for i in range(3)]

        # 7. Sort them from largest to smallest: major, minor2, minor1
        sorted_extents = np.sort(extents)[::-1]

        self.major_diameter = sorted_extents[0]
        self.minor_diameter2 = sorted_extents[1]
        self.minor_diameter1 = sorted_extents[2]

        # If you also want to store them labeled by axis:
        self.extent_x = extents[0]
        self.extent_y = extents[1]
        self.extent_z = extents[2]
        
        # As for radii (semi-axes), you can do e.g.:
        # (But be aware these "radii" might differ from the PCA-ellipsoid
        #  if you specifically want "closest to the axis" logic.)
        self.radii_custom = np.array(extents) / 2.0

    
    def fit_ellipsoid_scipy(self):
        def ellipsoid_error(params, positions):
            cx, cy, cz, a, b, c, alpha, beta, gamma = params
            rotation_matrix = self._create_rotation_matrix(alpha, beta, gamma)
            positions_centered = positions - np.array([cx, cy, cz])
            rotated_positions = positions_centered @ rotation_matrix.T
            x, y, z = rotated_positions.T
            distances = (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 - 1
            return distances

        def create_rotation_matrix(alpha, beta, gamma):
            # Create rotation matrix using Euler angles
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]
            ])
            Ry = np.array([
                [np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]
            ])
            Rz = np.array([
                [np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1]
            ])
            return Rz @ Ry @ Rx

        self._create_rotation_matrix = create_rotation_matrix

        node_ids = sorted(self.topology.nodes())
        valid_positions = np.array([self.positions[self.node_id_map[node_id]] for node_id in node_ids])
        centroid = valid_positions.mean(axis=0)
        initial_params = [*centroid, 1, 1, 1, 0, 0, 0]

        result = least_squares(ellipsoid_error, initial_params, args=(valid_positions,))
        cx, cy, cz, a, b, c, alpha, beta, gamma = result.x

        self.center = np.array([cx, cy, cz])
        self.radii = np.array([a, b, c])
        self.rotation_matrix = create_rotation_matrix(alpha, beta, gamma)
        
    def save_ellipsoid_plots(
        self,
        output_folder: Path,
        filename: str,
        sim_instance: MolecularDynamicsSimulation,
        *,
        dt_value: float,
        a_value: float,
        bsize_value: int,
        random_placement,
        random_chance,
        surface_triangulation: float,
        surface_area_fit: float,
        axis_major: float,
        axis_minor_1: float,
        axis_minor_2: float,
        axis_minor_mean: float,
        aspect_ratio: float,
        degree_counts: dict  # <-- NEW
    ):
        """
        Save a 2-row x 4-column plot:
        (Row 1) The ellipsoid and its 4 rotations
        (Row 2) EXACT 'SimulationAnimator' style static snapshot (same 4 rotations).

        Parameters:
            output_folder (Path): Directory to save the plots.
            filename (str): Name of the file (used for the plot filename).
            sim_instance (MolecularDynamicsSimulation): The simulation from which to grab the last state.
        """
        output_folder.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 4, subplot_kw={'projection': '3d'}, figsize=(20, 10))
        rotations = [(0, 0), (90, 0), (0, 90), (0, 0, 90)]

        # --------------------------
        # 1) TOP ROW: Ellipsoid
        # --------------------------
        for i, (elev, azim, *roll) in enumerate(rotations):
            ax = axes[0, i]

            # Plot positions
            x, y, z = self.positions[:, 0], self.positions[:, 1], self.positions[:, 2]
            ax.scatter(x, y, z, c='blue', s=10, label='Nodes')

            # Ellipsoid surface
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            xe = self.radii[0] * np.outer(np.cos(u), np.sin(v))
            ye = self.radii[1] * np.outer(np.sin(u), np.sin(v))
            ze = self.radii[2] * np.outer(np.ones_like(u), np.cos(v))

            # Rotate via PCA matrix
            xyz = np.array([xe.flatten(), ye.flatten(), ze.flatten()])
            rotated_xyz = self.rotation_matrix @ xyz
            xe_rot = rotated_xyz[0].reshape(xe.shape) + self.center[0]
            ye_rot = rotated_xyz[1].reshape(ye.shape) + self.center[1]
            ze_rot = rotated_xyz[2].reshape(ze.shape) + self.center[2]

            ax.plot_surface(xe_rot, ye_rot, ze_rot, rstride=4, cstride=4,
                            color='red', alpha=0.2)

            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Rotation {i + 1}')

        # --------------------------
        # 2) BOTTOM ROW: Animator
        # --------------------------
        last_state = sim_instance.state_trajectory[-1]
        topology = last_state['topology']
        node_ids = sorted(topology.nodes())
        node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        map_node_config = last_state.get('map_node_config', None)
        if map_node_config is None:
            map_node_config = sim_instance.map_node_config

        original_positions = last_state['positions']
        positions_plot = np.zeros((len(node_ids), 3, 3), dtype=original_positions.dtype)
        for i_node, node_id in enumerate(node_ids):
            original_index = map_node_config[node_id]
            positions_plot[i_node, :, :] = original_positions[original_index, :, :]

        plot_outer_layer = True
        if plot_outer_layer:
            vectors = positions_plot[:, 0, :]
        else:
            vectors = positions_plot.reshape(-1, 3)

        node_degrees = dict(topology.degree())
        colors = []
        if plot_outer_layer:
            for node_id in node_ids:
                deg = node_degrees.get(node_id, 0)
                if deg == 5:
                    colors.append((1, 0, 0, 1))          # red
                elif deg == 6:
                    colors.append((0.5, 0.5, 0.5, 0.1))  # gray alpha=0.1
                else:
                    colors.append((0, 0, 1, 1))          # blue
        else:
            for node_id in node_ids:
                deg = node_degrees.get(node_id, 0)
                if deg == 5:
                    color_tuple = (1, 0, 0, 1)
                elif deg == 6:
                    color_tuple = (0.5, 0.5, 0.5, 0.1)
                else:
                    color_tuple = (0, 0, 1, 1)
                colors.extend([color_tuple, color_tuple, color_tuple])

        scaling = sim_instance.monomer_info.get('scaling', 1.0)
        xs = vectors[:, 0] * scaling
        ys = vectors[:, 1] * scaling
        zs = vectors[:, 2] * scaling

        edges_list = list(topology.edges())

        def set_bounding_box(ax, xs, ys, zs):
            margin = 0
            x_min, x_max = xs.min() - margin, xs.max() + margin
            y_min, y_max = ys.min() - margin, ys.max() + margin
            z_min, z_max = zs.min() - margin, zs.max() + margin
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            z_center = (z_min + z_max) / 2.0

            ax.set_xlim(x_center - max_range / 2.0, x_center + max_range / 2.0)
            ax.set_ylim(y_center - max_range / 2.0, y_center + max_range / 2.0)
            ax.set_zlim(z_center - max_range / 2.0, z_center + max_range / 2.0)

        edge_alpha = 0.3
        for i, (elev, azim, *roll) in enumerate(rotations):
            ax = axes[1, i]

            ax.scatter(xs, ys, zs, s=20, c=colors, label='Particles')

            for edge in edges_list:
                n1, n2 = edge
                idx1 = node_id_to_index[n1]
                idx2 = node_id_to_index[n2]

                if plot_outer_layer:
                    layer_ids = [0]
                else:
                    layer_ids = [0, 1, 2]

                for layer_idx in layer_ids:
                    pos1 = positions_plot[idx1, layer_idx, :] * scaling
                    pos2 = positions_plot[idx2, layer_idx, :] * scaling
                    ax.plot(
                        [pos1[0], pos2[0]],
                        [pos1[1], pos2[1]],
                        [pos1[2], pos2[2]],
                        color='black',
                        alpha=edge_alpha
                    )

            set_bounding_box(ax, xs, ys, zs)
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Rotation {i + 1} (Animator Style)')

        # ------------------------------------------------------------------
        # 3) Add multi-line annotation with the parameters you requested
        # ------------------------------------------------------------------
        fig.subplots_adjust(top=0.80)  # Make a bit more space up top

        metadata_lines = [
            f"dt={dt_value}, a={a_value}, Bsize={bsize_value}",
            f"random_placement={random_placement}, random_chance={random_chance}",
            f"surface_triangulation={surface_triangulation}, surface_fit={surface_area_fit}",
            f"axis_major={axis_major}, axis_minor_1={axis_minor_1}, axis_minor_2={axis_minor_2}",
            f"axis_minor_mean={axis_minor_mean}, aspect_ratio={aspect_ratio}"
        ]
        # Build lines for node degrees only if > 1
        node_parts = []
        for d in range(1, 13):
            key = f"nodes_{d}"
            count = degree_counts[key]
            if count > 1:  # only show if there's more than 1
                node_parts.append(f"{key}={count}")

        # Combine node info into ONE line, e.g. "nodes_1=12 nodes_3=5"
        if node_parts:
            node_info_line = " ".join(node_parts)   # or ", ".join(node_parts) if you prefer commas
            metadata_lines.append(node_info_line)

        metadata_text = "\n".join(metadata_lines)

        plt.annotate(
            text=metadata_text,
            xy=(0.5, 0.98),
            xycoords='figure fraction',
            ha='center',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )

        plt.tight_layout()
        output_file = output_folder / f"{filename}.pdf"
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        return

if __name__ == '__main__':
    # Parameters
    input_folder = Path.cwd() / "simulations"
    output_folder = Path.cwd() / "sim_images"
    output_file = Path.cwd() / "extracted_sim_data.csv"
    max_degree_for_ellipsoid = 12  # Explicitly track degrees 1..12
    PLOT = True  # Set to True if you want to plot each simulation's ellipsoid

    # Regex patterns
    dt_pattern = re.compile(r"_dt([0-9.]+)")
    a_pattern = re.compile(r"_a([0-9.]+)")
    bsize_pattern = re.compile(r"_Bsize([0-9.]+)")

    # Collect data in a list
    data_records = []

    file_list = list(input_folder.resolve().glob("*.pkl"))

    # Initialize the tqdm progress bar
    for filepath in tqdm(file_list, desc="Processing files", unit="file"):
        try:
            filename_str = filepath.name
            
            # Load the simulation
            sim_instance = MolecularDynamicsSimulation.load_state(filepath)

            # Extract dt, a, Bsize from filename
            dt_match = dt_pattern.search(filename_str)
            a_match = a_pattern.search(filename_str)
            bsize_match = bsize_pattern.search(filename_str)

            if not (dt_match and a_match and bsize_match):
                #print(f"Skipping file {filename_str}: required parameters not found.")
                continue

            dt_value = float(dt_match.group(1))
            a_value = float(a_match.group(1))
            bsize_value = int(bsize_match.group(1))

            last_state = sim_instance.state_trajectory[-1]

            # Positions and topology
            positions = last_state['positions'][:, 0, :]
            topology = last_state['topology']

            # Initialize Ellipsoid and compute
            scaling = sim_instance.monomer_info['scaling']
            ellipsoid = Ellipsoid(positions, topology, scaling=scaling)
            ellipsoid.fit_ellipsoid_pca()
            #ellipsoid.fit_ellipsoid_scipy()
            #ellipsoid.fit_ellipsoid_pca_custom_diameters()
            surface_area_triangulation = ellipsoid.calc_surface_area(mode='triangulation')
            surface_area_fit = ellipsoid.calc_surface_area(mode='fit')
            


            # Compute axes
            axes_radii = np.array(ellipsoid.radii)
            idx_axis_major = np.argmax(axes_radii)
            axis_major = axes_radii[idx_axis_major]

            mask = np.ones(len(axes_radii), dtype=bool)
            mask[idx_axis_major] = False
            axes_minor = axes_radii[mask]
            axis_minor_1 = axes_minor[0] if len(axes_minor) > 0 else np.nan
            axis_minor_2 = axes_minor[1] if len(axes_minor) > 1 else np.nan
            axis_minor_mean = np.mean(axes_minor) if len(axes_minor) > 0 else np.nan

            # Aspect ratio
            if axis_minor_mean != 0 and not np.isnan(axis_minor_mean):
                aspect_ratio = axis_major / axis_minor_mean
            else:
                aspect_ratio = np.nan

            # Node degrees 1..12
            degree_counts = {f"nodes_{d}": 0 for d in range(1, max_degree_for_ellipsoid + 1)}
            for node in topology.nodes():
                deg = topology.degree(node)
                if deg < max_degree_for_ellipsoid:
                    degree_counts[f"nodes_{deg}"] += 1
                else:
                    degree_counts[f"nodes_{max_degree_for_ellipsoid}"] += 1

            # Also extract random_placement and random_chance
            random_placement = getattr(sim_instance, 'random_placement', None)
            random_chance = getattr(sim_instance, 'random_chance', None)

            ## Filter for valid containers:
            cond_1 = sim_instance.is_closed_surface()
            cond_2 = degree_counts["nodes_4"] <= 10 #(degree_counts["nodes_4"]/degree_counts["nodes_6"]) <= 0.07 # Sometimes the is_closed_surface is not sufficient. But in these cases, the fraction of Nodes of degree 4 is very high.
            valid_container = cond_1 and cond_2
            
            if PLOT and a_value  <= 10 and valid_container:
                ellipsoid.save_ellipsoid_plots(
                    output_folder=output_folder,
                    filename=filename_str,
                    sim_instance=sim_instance,
                    dt_value=dt_value,
                    a_value=a_value,
                    bsize_value=bsize_value,
                    random_placement=random_placement,
                    random_chance=random_chance,
                    surface_triangulation=surface_area_triangulation,
                    surface_area_fit=surface_area_fit,
                    axis_major=axis_major,
                    axis_minor_1=axis_minor_1,
                    axis_minor_2=axis_minor_2,
                    axis_minor_mean=axis_minor_mean,
                    aspect_ratio=aspect_ratio,
                    degree_counts=degree_counts
                )
                
                
            if valid_container:         
                # Build row
                row = {
                    "filename": filename_str,
                    "dt": dt_value,
                    "a": a_value,
                    "Bsize": bsize_value,
                    "random_placement": random_placement,
                    "random_chance": random_chance,
                    "surface_triangulation": surface_area_triangulation,
                    "surface_fit": surface_area_fit,
                    "axis_major": axis_major,
                    "axis_minor_1": axis_minor_1,
                    "axis_minor_2": axis_minor_2,
                    "axis_minor_mean": axis_minor_mean,
                    "aspect_ratio": aspect_ratio,
                    "closed_surface": True,
                }
                row.update(degree_counts)

                data_records.append(row)
            else:         
                # Build row
                row = {
                    "filename": filename_str,
                    "dt": dt_value,
                    "a": a_value,
                    "Bsize": bsize_value,
                    "random_placement": random_placement,
                    "random_chance": random_chance,
                    "surface_triangulation": surface_area_triangulation,
                    "surface_fit": surface_area_fit,
                    "axis_major": axis_major,
                    "axis_minor_1": axis_minor_1,
                    "axis_minor_2": axis_minor_2,
                    "axis_minor_mean": axis_minor_mean,
                    "aspect_ratio": aspect_ratio,
                    "closed_surface": False,
                }
                row.update(degree_counts)

                data_records.append(row)
        except Exception as e:
            #print(f"Skipping {filepath.name} due to exception: {e}")
            continue


    # Create DataFrame
    df = pd.DataFrame(data_records)

    # Sort by dt, a, Bsize
    sorted_df = df.sort_values(by=["dt", "a", "Bsize"], ascending=[True, True, True])

    # Save to CSV
    sorted_df.to_csv(output_file, index=False)
    #df.to_csv(output_file, index=False)