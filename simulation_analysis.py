import math
import numpy as np
import time
import traceback
import pickle
import copy
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from scipy.spatial import KDTree

class MolecularDynamicsSimulation:
    def __init__(self, dt, mass, lengthEq, delta, km, T_C=20, origin=np.zeros(3), method='verlet', damping_coeff=100):
        str_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = str_datetime + '_sim_' + method + "_dt" + str(dt) + "_delta" + str(delta) + "_km" + str(km)+'_TC' + str(T_C)
        if method == 'langevin':
            filename += "_damping" + str(damping_coeff)
        filename += '.pkl'
        self.filename = filename

        self.state_trajectory = []
        self.sim_trajectory = []

        self.dt = np.float64(dt)
        self.mass = np.float64(mass)

        self.k1 = np.float64(km)
        self.k2 = np.float64(km / 2)
        self.k3 = np.float64(km / 3)

        self.lengthEq = np.float64(lengthEq)
        self.delta = np.float64(delta)

        self.T_C = T_C
        self.T_K = T_C + 273.15

        self.origin = origin

        self.method = method
        self.damping_coeff = damping_coeff

        self.topology = nx.Graph()

        # Holds the initial particles constituting the first unit cell (a triangular prism)
        initial_positions = self.create_monomer()

        # Map node IDs to indices
        self.node_ids = []
        self.node_id_map = {}
        self.N_nodes = 0  # Will be updated as we add nodes

        # Initialize positions, velocities, accelerations, positions_old arrays
        # Shape: (N_nodes, 3_layers, 3)
        self.positions = np.zeros((0, 3, 3), dtype=np.float64)
        self.velocities = np.zeros((0, 3, 3), dtype=np.float64)
        self.accelerations = np.zeros((0, 3, 3), dtype=np.float64)
        self.positions_old = np.zeros((0, 3, 3), dtype=np.float64)

        # Add initial particles
        for idx, pos_layers in enumerate(initial_positions):
            node_id = idx  # Assuming initial node IDs start from 0
            self.add_particle(node_id, pos_layers)

        self.fix_velocity_distribution()

        # Empirically determine the equilibrium cross-layer distance
        # Distance between layer 1 and 2 -> a12
        # Distance between layer 2 and 3 -> a23
        self.a12 = np.linalg.norm(self.positions[0, 0, :] - self.positions[1, 1, :])
        self.a23 = np.linalg.norm(self.positions[0, 1, :] - self.positions[1, 2, :])

        # Generate a complete topological graph from initial configuration
        complete_graph = nx.complete_graph(self.topology.nodes())
        self.topology.add_edges_from(complete_graph.edges())

        # Initialize boundary edges
        self.boundary_edges = set()
        for edge in self.topology.edges():
            self.boundary_edges.add(tuple(sorted(edge)))
            
        print(f'Initial boundary: {self.boundary_edges}')

    # Helper functions
    def create_monomer(self):
        l1 = self.lengthEq + 2 * self.delta
        l2 = self.lengthEq
        l3 = self.lengthEq - 2 * self.delta

        ori_X = self.origin[0]
        ori_Y = self.origin[1]
        ori_Z = self.origin[2]

        h1 = l1 * math.sqrt(3) / 2
        h2 = l2 * math.sqrt(3) / 2
        h3 = l3 * math.sqrt(3) / 2

        u = np.linalg.norm(np.array([l1 / 2 - l2 / 2, h1 / 3 - h2 / 3], dtype=np.float64))
        dz = math.sqrt(self.lengthEq ** 2 - u ** 2)

        # Positions for each layer
        # Layer 0 (Top)
        pos_0 = np.array([
            [ori_X - l1 / 2, ori_Y - h1 / 3, ori_Z],
            [ori_X + l1 / 2, ori_Y - h1 / 3, ori_Z],
            [ori_X, ori_Y + h1 * 2 / 3, ori_Z]
        ], dtype=np.float64)

        # Layer 1 (Middle)
        pos_1 = np.array([
            [ori_X - l2 / 2, ori_Y - h2 / 3, ori_Z + dz],
            [ori_X + l2 / 2, ori_Y - h2 / 3, ori_Z + dz],
            [ori_X, ori_Y + h2 * 2 / 3, ori_Z + dz]
        ], dtype=np.float64)

        # Layer 2 (Bottom)
        pos_2 = np.array([
            [ori_X - l3 / 2, ori_Y - h3 / 3, ori_Z + 2 * dz],
            [ori_X + l3 / 2, ori_Y - h3 / 3, ori_Z + 2 * dz],
            [ori_X, ori_Y + h3 * 2 / 3, ori_Z + 2 * dz]
        ], dtype=np.float64)

        # Combine positions into a list of positions per node
        positions = []
        for i in range(3):  # Three nodes
            pos_layers = np.array([pos_0[i], pos_1[i], pos_2[i]], dtype=np.float64)
            positions.append(pos_layers)

        return positions

    def mirror_point(self, A, B, C):
        # Mirror point B across the line AC
        AC = C - A
        AB = B - A
        AC_norm = AC / np.linalg.norm(AC)
        projection_length = np.dot(AB, AC_norm)
        projection_point = A + projection_length * AC_norm
        mirrored_B = 2 * projection_point - B
        return mirrored_B

    def calc_angle(self, node_id, neighbour1_id, neighbour2_id):
        pos_node = self.positions[self.node_id_map[node_id], 0, :]
        pos_neighbour1 = self.positions[self.node_id_map[neighbour1_id], 0, :]
        pos_neighbour2 = self.positions[self.node_id_map[neighbour2_id], 0, :]

        v1 = pos_neighbour1 - pos_node
        v2 = pos_neighbour2 - pos_node

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot_product / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        angle_degrees = np.degrees(angle)

        return angle_degrees

    def calc_force_batch(self, v_r1_array, v_r2_array, k, a0):
        v_r12 = v_r1_array - v_r2_array  # Shape: (N, 3)
        l = np.linalg.norm(v_r12, axis=1)  # Shape: (N,)

        # Prevent division by zero
        l_safe = np.where(l == 0, 1, l)

        v_r12_norm = v_r12 / l_safe[:, np.newaxis]  # Shape: (N, 3)
        # Adjust force where l == 0 to zero
        v_r12_norm[l == 0] = 0

        v_f = -k * (l - a0)[:, np.newaxis] * v_r12_norm  # Shape: (N, 3)
        return v_f

    # Getters
    def getParticleCount(self):
        return self.topology.number_of_nodes()

    # Check if there is still a boundary in the assembly -> if not, the surface must be closed
    def is_closed_surface(self):
        return len(self.boundary_edges) <= 3 and self.getParticleCount() > 3

    def update_accelerations(self):
        # Reset accelerations
        self.accelerations.fill(0)  # Shape: (N_nodes, 3, 3)

        # Extract positions for each layer
        pos_top = self.positions[:, 0, :]  # Shape: (N_nodes, 3)
        pos_mid = self.positions[:, 1, :]
        pos_low = self.positions[:, 2, :]

        # Self-interactions (forces between layers within the same node)
        forces_top_mid_self = self.calc_force_batch(pos_top, pos_mid, self.k2, self.lengthEq)
        forces_mid_low_self = self.calc_force_batch(pos_mid, pos_low, self.k2, self.lengthEq)

        # Update accelerations for self-interactions
        self.accelerations[:, 0, :] += forces_top_mid_self / self.mass
        self.accelerations[:, 1, :] += (-forces_top_mid_self + forces_mid_low_self) / self.mass
        self.accelerations[:, 2, :] += -forces_mid_low_self / self.mass

        # Build interaction lists for neighbor interactions
        node_indices = []
        neighbor_indices = []

        for edge in self.topology.edges():
            node_id = edge[0]
            neighbour_id = edge[1]
            node_indices.append(self.node_id_map[node_id])
            neighbor_indices.append(self.node_id_map[neighbour_id])

        node_indices = np.array(node_indices)
        neighbor_indices = np.array(neighbor_indices)

        # Positions of interacting pairs
        pos_node_layers = self.positions[node_indices, :, :]  # Shape: (N_pairs, 3, 3)
        pos_neigh_layers = self.positions[neighbor_indices, :, :]

        # Same-layer forces (k1 interactions)
        force_top = self.calc_force_batch(pos_node_layers[:, 0, :], pos_neigh_layers[:, 0, :],
                                          self.k1, self.lengthEq + 2 * self.delta)
        force_mid = self.calc_force_batch(pos_node_layers[:, 1, :], pos_neigh_layers[:, 1, :],
                                          self.k1, self.lengthEq)
        force_low = self.calc_force_batch(pos_node_layers[:, 2, :], pos_neigh_layers[:, 2, :],
                                          self.k1, self.lengthEq - 2 * self.delta)

        # Cross-layer forces (k3 interactions)
        force_top_mid = self.calc_force_batch(pos_node_layers[:, 0, :], pos_neigh_layers[:, 1, :],
                                              self.k3, self.a12)
        force_mid_top = self.calc_force_batch(pos_node_layers[:, 1, :], pos_neigh_layers[:, 0, :],
                                              self.k3, self.a12)
        force_mid_low = self.calc_force_batch(pos_node_layers[:, 1, :], pos_neigh_layers[:, 2, :],
                                              self.k3, self.a23)
        force_low_mid = self.calc_force_batch(pos_node_layers[:, 2, :], pos_neigh_layers[:, 1, :],
                                              self.k3, self.a23)

        # Update accelerations using np.add.at to handle multiple contributions
        # Same-layer interactions
        np.add.at(self.accelerations, (node_indices, 0, slice(None)), force_top / self.mass)
        np.add.at(self.accelerations, (neighbor_indices, 0, slice(None)), -force_top / self.mass)

        np.add.at(self.accelerations, (node_indices, 1, slice(None)), force_mid / self.mass)
        np.add.at(self.accelerations, (neighbor_indices, 1, slice(None)), -force_mid / self.mass)

        np.add.at(self.accelerations, (node_indices, 2, slice(None)), force_low / self.mass)
        np.add.at(self.accelerations, (neighbor_indices, 2, slice(None)), -force_low / self.mass)

        # Cross-layer interactions
        # Top to mid
        np.add.at(self.accelerations, (node_indices, 0, slice(None)), force_top_mid / self.mass)
        np.add.at(self.accelerations, (neighbor_indices, 1, slice(None)), -force_top_mid / self.mass)

        # Mid to top
        np.add.at(self.accelerations, (node_indices, 1, slice(None)), force_mid_top / self.mass)
        np.add.at(self.accelerations, (neighbor_indices, 0, slice(None)), -force_mid_top / self.mass)

        # Mid to low
        np.add.at(self.accelerations, (node_indices, 1, slice(None)), force_mid_low / self.mass)
        np.add.at(self.accelerations, (neighbor_indices, 2, slice(None)), -force_mid_low / self.mass)

        # Low to mid
        np.add.at(self.accelerations, (node_indices, 2, slice(None)), force_low_mid / self.mass)
        np.add.at(self.accelerations, (neighbor_indices, 1, slice(None)), -force_low_mid / self.mass)

    def calcTotalEnergy(self):
        totalEnergy = 0

        # Build interaction lists for neighbor interactions
        node_indices = []
        neighbor_indices = []

        for edge in self.topology.edges():
            node_id = edge[0]
            neighbour_id = edge[1]
            node_indices.append(self.node_id_map[node_id])
            neighbor_indices.append(self.node_id_map[neighbour_id])

        node_indices = np.array(node_indices)
        neighbor_indices = np.array(neighbor_indices)

        # Positions of interacting pairs
        pos_node_layers = self.positions[node_indices, :, :]  # Shape: (N_pairs, 3, 3)
        pos_neigh_layers = self.positions[neighbor_indices, :, :]

        # Same-layer potential energy (k1 interactions)
        energy_top = 0.5 * self.k1 * (np.linalg.norm(pos_node_layers[:, 0, :] - pos_neigh_layers[:, 0, :], axis=1) -
                                      (self.lengthEq + 2 * self.delta)) ** 2
        energy_mid = 0.5 * self.k1 * (np.linalg.norm(pos_node_layers[:, 1, :] - pos_neigh_layers[:, 1, :], axis=1) -
                                      self.lengthEq) ** 2
        energy_low = 0.5 * self.k1 * (np.linalg.norm(pos_node_layers[:, 2, :] - pos_neigh_layers[:, 2, :], axis=1) -
                                      (self.lengthEq - 2 * self.delta)) ** 2

        # Cross-layer potential energy (k3 interactions)
        energy_top_mid = 0.5 * self.k3 * (np.linalg.norm(pos_node_layers[:, 0, :] - pos_neigh_layers[:, 1, :], axis=1) -
                                          self.a12) ** 2
        energy_mid_top = 0.5 * self.k3 * (np.linalg.norm(pos_node_layers[:, 1, :] - pos_neigh_layers[:, 0, :], axis=1) -
                                          self.a12) ** 2
        energy_mid_low = 0.5 * self.k3 * (np.linalg.norm(pos_node_layers[:, 1, :] - pos_neigh_layers[:, 2, :], axis=1) -
                                          self.a23) ** 2
        energy_low_mid = 0.5 * self.k3 * (np.linalg.norm(pos_node_layers[:, 2, :] - pos_neigh_layers[:, 1, :], axis=1) -
                                          self.a23) ** 2

        totalEnergy += np.sum(energy_top + energy_mid + energy_low +
                              energy_top_mid + energy_mid_top + energy_mid_low + energy_low_mid)

        # Self-interactions (forces between layers within the same node)
        pos_top = self.positions[:, 0, :]  # Shape: (N_nodes, 3)
        pos_mid = self.positions[:, 1, :]
        pos_low = self.positions[:, 2, :]

        energy_top_mid_self = 0.5 * self.k2 * (np.linalg.norm(pos_top - pos_mid, axis=1) - self.lengthEq) ** 2
        energy_mid_low_self = 0.5 * self.k2 * (np.linalg.norm(pos_mid - pos_low, axis=1) - self.lengthEq) ** 2

        totalEnergy += np.sum(energy_top_mid_self + energy_mid_low_self)

        return totalEnergy

    def add_particle(self, node_id, pos_layers):
        # Add a new particle with given positions for each layer
        self.topology.add_node(node_id)
        self.node_id_map[node_id] = self.N_nodes
        self.node_ids.append(node_id)
        self.N_nodes += 1

        # Expand arrays to accommodate the new node
        self.positions = np.vstack((self.positions, [pos_layers]))
        self.velocities = np.vstack((self.velocities, [self.init_vel_boltzmann(3)]))
        self.accelerations = np.vstack((self.accelerations, [np.zeros((3, 3), dtype=np.float64)]))
        self.positions_old = np.vstack((self.positions_old, [pos_layers - self.velocities[-1] * self.dt]))

    def addParticle(self):
        # Use boundary edges to find nodes at the boundary
        boundary_nodes = set()
        for edge in self.boundary_edges:
            boundary_nodes.update(edge)

        # Remove any nodes that are no longer in the topology
        boundary_nodes = {node for node in boundary_nodes if node in self.node_id_map}

        angles = []

        # Iterate through all nodes at the boundary
        for node in boundary_nodes:
            degree = self.topology.degree(node)
            connected_edges = [edge for edge in self.boundary_edges if node in edge]
            connected_ids = [edge[1] if edge[0] == node else edge[0] for edge in connected_edges]
            if len(connected_ids) < 2:
                continue  # Skip nodes that don't have at least two boundary connections
            angle = self.calc_angle(node, connected_ids[0], connected_ids[1])
            if degree == 2 or degree == 3:
                angle = 360 - angle

            entry = {'angle': angle, 'node_id': node, 'degree': degree,
                     'neighbour_1': connected_ids[0], 'neighbour_2': connected_ids[1]}
            angles.append(entry)

        if not angles:
            print("No valid nodes to add particles.")
            return

        min_entry = min(angles, key=lambda x: x['angle'])

        if min_entry['angle'] < 30:
            print("CLOSE AS PENTAMER")

            # Remove the two edges connected to neighbour_2 from boundary_edges
            edge1 = tuple(sorted((min_entry['node_id'], min_entry['neighbour_1'])))
            edge2 = tuple(sorted((min_entry['node_id'], min_entry['neighbour_2'])))
            self.boundary_edges.discard(edge1)
            self.boundary_edges.discard(edge2)

            # Remove boundary edges involving neighbour_2
            edges_to_remove = {edge for edge in self.boundary_edges if min_entry['neighbour_2'] in edge}
            self.boundary_edges -= edges_to_remove

            new_boundaries = {tuple(sorted((node, min_entry['neighbour_1']))) for edge in edges_to_remove for node in edge if node != min_entry['neighbour_2']}
            self.boundary_edges.update(new_boundaries)

            # Add new edges and update boundary_edges
            for neighbour_id in list(self.topology.neighbors(min_entry['neighbour_2'])):
                self.topology.add_edge(min_entry['neighbour_1'], neighbour_id)
                    
            # Remove node and update data structures
            self.topology.remove_node(min_entry['neighbour_2'])
            idx_remove = self.node_id_map.pop(min_entry['neighbour_2'])
            self.node_ids.remove(min_entry['neighbour_2'])
            self.N_nodes -= 1
            self.positions = np.delete(self.positions, idx_remove, axis=0)
            self.velocities = np.delete(self.velocities, idx_remove, axis=0)
            self.accelerations = np.delete(self.accelerations, idx_remove, axis=0)
            self.positions_old = np.delete(self.positions_old, idx_remove, axis=0)

            # Update node_id_map indices
            for node_id, idx in self.node_id_map.items():
                if idx > idx_remove:
                    self.node_id_map[node_id] = idx - 1

        elif 30 <= min_entry['angle'] <= 60 and min_entry['degree'] == 6:
            print("CLOSE AS HEXAMER")
            # Close as Hexamer
            edge1 = tuple(sorted((min_entry['node_id'], min_entry['neighbour_1'])))
            edge2 = tuple(sorted((min_entry['node_id'], min_entry['neighbour_2'])))
            self.boundary_edges.discard(edge1)
            self.boundary_edges.discard(edge2)

            # Add new edge and update boundary_edges
            self.topology.add_edge(min_entry['neighbour_1'], min_entry['neighbour_2'])
            new_edge = tuple(sorted((min_entry['neighbour_1'], min_entry['neighbour_2'])))
            if new_edge in self.boundary_edges:
                self.boundary_edges.discard(new_edge)
            else:
                self.boundary_edges.add(new_edge)
        else:
            # No pentamer or hexamer closure: add new particle
            print(f"ADD NEW PARTICLE (TOTAL_NUM_PARTICLES={self.getParticleCount()})")

            node_id = max(self.node_ids) + 1  # Generate new node ID

            self.placeParticle(node_id, min_entry['node_id'], min_entry['neighbour_1'])

            # Update boundary edges
            edge_to_remove = tuple(sorted((min_entry['node_id'], min_entry['neighbour_1'])))
            self.boundary_edges.discard(edge_to_remove)

            # Add new edges to boundary
            new_edge1 = tuple(sorted((node_id, min_entry['node_id'])))
            new_edge2 = tuple(sorted((node_id, min_entry['neighbour_1'])))
            self.boundary_edges.add(new_edge1)
            self.boundary_edges.add(new_edge2)

    def placeParticle(self, id, edgeStart, edgeEnd):
        anchor_nodes = list(nx.common_neighbors(self.topology, edgeStart, edgeEnd))
        if not anchor_nodes:
            print("No common neighbor found.")
            return
        anchor_id = anchor_nodes[0]
        anchor_idx = self.node_id_map[anchor_id]
        edgeStart_idx = self.node_id_map[edgeStart]
        edgeEnd_idx = self.node_id_map[edgeEnd]

        pos_anchor = self.positions[anchor_idx, :, :]  # Shape: (3, 3)
        pos_edgeStart = self.positions[edgeStart_idx, :, :]
        pos_edgeEnd = self.positions[edgeEnd_idx, :, :]

        pos_new_layers = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            pAnchor = pos_anchor[i, :]
            pBase11 = pos_edgeStart[i, :]
            pBase21 = pos_edgeEnd[i, :]

            coords = self.mirror_point(pBase11, pAnchor, pBase21)
            pos_new_layers[i, :] = coords

        self.add_particle(id, pos_new_layers)
        self.fix_velocity_distribution()

        self.topology.add_node(id)
        self.topology.add_edge(id, edgeStart)
        self.topology.add_edge(id, edgeEnd)

    def fix_topology(self, min_topo_dist=5, max_physical_dist=1.0, check_interval=50, current_step=0):
        # Only check every X steps
        if current_step % check_interval != 0:
            return

        # Extract positions of nodes and create a KDTree for efficient spatial searching
        indices = {idx: node_id for node_id, idx in self.node_id_map.items()}
        positions = self.positions[:, 0, :]  # Use the top layer positions for proximity check

        kd_tree = KDTree(positions)

        # For each node, find nearby nodes (within a given distance)
        boundary_nodes = set()
        for edge in self.boundary_edges:
            boundary_nodes.update(edge)

        # Remove any nodes that are no longer in the topology
        boundary_nodes = {node for node in boundary_nodes if node in self.node_id_map}

        for node in boundary_nodes.copy():
            node_idx = self.node_id_map[node]
            node_pos = self.positions[node_idx, 0, :]
            results = kd_tree.query_ball_point(node_pos, r=max_physical_dist)

            node_ids = [indices[result] for result in results if indices[result] != node]
            for other_node in node_ids:
                # Check if other_node is still valid
                if other_node not in self.node_id_map:
                    continue
                # Check topological distance
                if nx.has_path(self.topology, node, other_node):
                    topo_dist = nx.shortest_path_length(self.topology, source=node, target=other_node)

                    if topo_dist >= min_topo_dist:
                        print('FIXING EVENT OCCURRED')

                        # Merge nodes if criteria are met
                        for neighbour_id in list(self.topology.neighbors(other_node)):
                            self.topology.add_edge(node, neighbour_id)
                            new_edge = tuple(sorted((node, neighbour_id)))
                            if new_edge in self.boundary_edges:
                                self.boundary_edges.discard(new_edge)
                            else:
                                self.boundary_edges.add(new_edge)
                        # Remove boundary edges involving other_node
                        edges_to_remove = {edge for edge in self.boundary_edges if other_node in edge}
                        self.boundary_edges -= edges_to_remove

                        self.topology.remove_node(other_node)
                        idx_remove = self.node_id_map.pop(other_node)
                        self.node_ids.remove(other_node)
                        self.N_nodes -= 1
                        # Remove positions, velocities, etc., corresponding to the removed node
                        self.positions = np.delete(self.positions, idx_remove, axis=0)
                        self.velocities = np.delete(self.velocities, idx_remove, axis=0)
                        self.accelerations = np.delete(self.accelerations, idx_remove, axis=0)
                        self.positions_old = np.delete(self.positions_old, idx_remove, axis=0)

                        # Update node_id_map indices
                        for node_id, idx in self.node_id_map.items():
                            if idx > idx_remove:
                                self.node_id_map[node_id] = idx - 1
                        return

    def fix_cycles(self):
        if self.getParticleCount() == 3:
            return
        
        G = nx.Graph()
        G.add_edges_from(self.boundary_edges)
        cycles = nx.cycle_basis(G)
        
        # Filter for triangular cycles (cycles with exactly 3 nodes)
        triangular_cycles = [cycle for cycle in cycles if len(cycle) == 3]

        # Convert each triangular cycle into edges and remove from original edges
        for cycle in triangular_cycles:
            # Convert the cycle nodes to edges
            cycle_edges = {tuple(sorted((cycle[i], cycle[(i+1) % 3]))) for i in range(3)}
            # Remove these edges from the original set
            self.boundary_edges -= cycle_edges
        

    def init_vel_boltzmann(self, num_particles):
        k_B = 1.38e-23
        sigma = np.sqrt(k_B * self.T_K / self.mass)
        vel = np.random.normal(0, sigma, (num_particles, 3)).astype(np.float64)
        return vel

    def fix_velocity_distribution(self):
        k_B = 1.38e-23
        N = self.N_nodes * 3  # Total number of particles (nodes * layers)

        # Flatten velocities to shape (N, 3)
        vel_flat = self.velocities.reshape(-1, 3)

        # Step 1: Calculate the total velocity (sum of all velocities)
        v_total = np.sum(vel_flat, axis=0)

        # Step 2: Calculate the mean velocity
        v_mean = v_total / N

        # Step 3: Subtract the mean velocity from each particle
        vel_flat -= v_mean

        # Step 4: Calculate the total squared velocity
        v_square_total = np.sum(vel_flat ** 2)

        # Step 5: Calculate the mean squared velocity (average over all particles)
        v_square_mean = v_square_total / N

        # Step 6: Calculate the current kinetic energy and the desired kinetic energy
        kinetic_energy_per_particle = 0.5 * self.mass * v_square_mean
        desired_kinetic_energy = 1.5 * k_B * self.T_K

        # Step 7: Compute the scaling factor
        scaling_factor = np.sqrt(desired_kinetic_energy / kinetic_energy_per_particle)

        # Step 8: Rescale each particle's velocity
        vel_flat *= scaling_factor

        # Reshape velocities back to original shape
        self.velocities = vel_flat.reshape(self.N_nodes, 3, 3)

    # Load and save
    def save_state(self):
        cd = Path(__file__).parent.resolve()

        # Prepare the new state
        state = {
            'positions': copy.deepcopy(self.positions),
            'topology': copy.deepcopy(self.topology),
            'positions_old': copy.deepcopy(self.positions_old),
            'velocities': copy.deepcopy(self.velocities),
            'accelerations': copy.deepcopy(self.accelerations)
        }
        self.state_trajectory.append(state)

        # Only pickle self
        with open(cd / self.filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_state(cls, filename, filename_append='', start_at=-1):
        with open(filename, 'rb') as f:
            loaded_instance = pickle.load(f)

        trajectory = copy.deepcopy(loaded_instance.state_trajectory[start_at])

        # Update the filename
        if filename_append:
            loaded_instance.filename = '.'.join(loaded_instance.filename.split('.')[:-1]) + '_' + filename_append + '.pkl'

        # Update the state
        loaded_instance.positions = trajectory['positions']
        loaded_instance.positions_old = trajectory['positions_old']
        loaded_instance.velocities = trajectory['velocities']
        loaded_instance.accelerations = trajectory['accelerations']
        loaded_instance.topology = trajectory['topology']

        return loaded_instance

    # Simulation
    def verlet_update(self):
        # Recalculate forces and update accelerations
        self.update_accelerations()

        # Update positions using Verlet algorithm
        dt2 = self.dt ** 2
        positions_new = 2 * self.positions - self.positions_old + self.accelerations * dt2

        # Update old positions
        self.positions_old[:] = self.positions.copy()

        # Update positions
        self.positions[:] = positions_new

    def langevin_update(self):
        # Recalculate forces and update accelerations
        self.update_accelerations()

        # Update positions using Langevin algorithm
        positions_new = self.positions + (self.accelerations * self.mass * self.dt) / self.damping_coeff

        # Update old positions
        self.positions_old[:] = self.positions.copy()

        # Update positions
        self.positions[:] = positions_new

    def simulate_step(self):
        if self.method == 'verlet':
            self.verlet_update()
        elif self.method == 'langevin':
            self.langevin_update()
        else:
            raise Exception("No valid method chosen")


class SimulationAnimator:
    def __init__(self, sim_instance, plot_outer_layer=True, loop=False):
        self.sim_instance = sim_instance
        self.plot_outer_layer = plot_outer_layer
        self.loop = loop
        self.state_trajectory = sim_instance.state_trajectory
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1, 1, 1])
        # Initialize other necessary attributes
        # Initialize the animation
        self.initialize_animation()

    def initialize_animation(self):
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Capsid Assembly Simulation')

        # Initialize scatter plot
        self.scatter = self.ax.scatter([], [], [], s=20, label='Particles')
        # Initialize edge lines
        # Compute maximum number of edges to allocate line objects
        max_edges = max(len(state['topology'].edges()) for state in self.state_trajectory)
        self.edge_lines = []
        for _ in range(max_edges * (1 if self.plot_outer_layer else 3)):
            line, = self.ax.plot([], [], [], color='black')
            self.edge_lines.append(line)
        
        # Animation control flags
        self.anim_running = True
        self.current_frame = 0

        # Set up key event handlers
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Create a timer explicitly
        self.timer = self.fig.canvas.new_timer(interval=20)
        self.timer.add_callback(self._update_animation)

        # Create the animation with the explicit timer
        self.animation = FuncAnimation(self.fig, self.update, frames=len(self.state_trajectory), 
                                       interval=5, blit=False, repeat=self.loop, event_source=self.timer)
        self.timer.start()
        plt.show()

    def on_key_press(self, event):
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
        # Reconstruct node_id_map
        node_ids = sorted(topology.nodes())
        node_id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
        N_nodes = positions.shape[0]
        if self.plot_outer_layer:
            vectors = positions[:, 0, :]
        else:
            vectors = positions.reshape(-1, 3)
        xs = vectors[:, 0]
        ys = vectors[:, 1]
        zs = vectors[:, 2]
        # Colors based on node degree
        node_degrees = dict(topology.degree())
        colors = []
        if self.plot_outer_layer:
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
                for _ in range(3):
                    if degree == 5:
                        colors.append('red')
                    elif degree == 6:
                        colors.append('blue')
                    else:
                        colors.append('gray')
        self.scatter._offsets3d = (xs, ys, zs)
        self.scatter.set_color(colors)
        # Clear previous edges
        for line in self.edge_lines:
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
            if self.plot_outer_layer:
                layers = [0]
            else:
                layers = [0, 1, 2]
            for layer in layers:
                pos1 = positions[idx1, layer, :]
                pos2 = positions[idx2, layer, :]
                line = self.edge_lines[edge_count]
                line.set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
                line.set_3d_properties([pos1[2], pos2[2]])
                edge_count += 1
        # Hide unused edge lines
        for i in range(edge_count, len(self.edge_lines)):
            line = self.edge_lines[i]
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
        self.ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
        self.ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
        self.ax.set_zlim(z_center - max_range / 2, z_center + max_range / 2)
        plt.draw()
        return self.scatter, *self.edge_lines
    
if __name__ == '__main__':
    # Load and animate the trajectory
    #filename = '20241115114345_sim_langevin_dt0.05_delta0.15_km1_TC20_damping0.5.pkl'
    filename = '20241115133330_sim_langevin_dt0.01_delta0.1_km0.1_TC20_damping0.1_cont1_cont1.pkl'  # Replace with your actual filename
    cd = Path().resolve()
    filepath = cd / filename

    sim_instance = MolecularDynamicsSimulation.load_state(filepath)
    sim_instance.state_trajectory = sim_instance.state_trajectory[-50:]

    animator = SimulationAnimator(sim_instance, plot_outer_layer=True, loop=False)

    # Print node degrees at the last frame
    last_state = sim_instance.state_trajectory[-1]
    topology = last_state['topology']

    degree_array = np.zeros(8, dtype=int)
    for node in topology.nodes():
        degree = topology.degree(node)
        if degree < len(degree_array):
            degree_array[degree] += 1
        else:
            # Extend the array if necessary
            degree_array = np.resize(degree_array, degree + 1)
            degree_array[degree] = 1

    for i in range(len(degree_array)):
        print(f'DEGREE {i}: {degree_array[i]} NODES')
