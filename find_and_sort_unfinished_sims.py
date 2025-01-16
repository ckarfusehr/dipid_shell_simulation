import re
import shutil
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from simulation_assembly import MolecularDynamicsSimulation

def count_node_degrees(topology, max_degree=11):
    degree_array = np.zeros(max_degree + 2, dtype=int)
    for node in topology.nodes():
        degree = topology.degree(node)
        if degree <= max_degree:
            degree_array[degree] += 1
        else:
            degree_array[-1] += 1
    return degree_array

def main():
    input_folder = Path("simulations")
    unfinished_folder = Path("unfinished_sims")
    unfinished_folder.mkdir(exist_ok=True)

    a_min = 0.0
    a_max = 10.0
    acceptable_random_chance = [0.0, 0.05]

    dt_pattern = re.compile(r"_dt([0-9.]+)")
    a_pattern = re.compile(r"_a([0-9.]+)")
    bsize_pattern = re.compile(r"_Bsize([0-9.]+)")

    files = list(input_folder.glob("*.pkl"))
    for filepath in tqdm(files, desc="Processing simulations"):
        filename = filepath.name

        dt_match = dt_pattern.search(filename)
        a_match = a_pattern.search(filename)
        bsize_match = bsize_pattern.search(filename)

        if not (dt_match and a_match and bsize_match):
            print(f"Skipping file {filename}: required parameters not found.")
            continue

        a_value = float(a_match.group(1))
        sim_instance = MolecularDynamicsSimulation.load_state(filepath)

        if (
            a_min <= a_value <= a_max 
            and sim_instance.is_closed_surface() is False
            and sim_instance.random_chance in acceptable_random_chance
        ):
            last_state = sim_instance.state_trajectory[-1]
            topology = last_state["topology"]
            degree_array = count_node_degrees(topology)

            if degree_array[6] < 3000:
                target_path = unfinished_folder / filename
                print(f"Moving {filename} to {target_path}")
                shutil.move(str(filepath), str(target_path))

if __name__ == "__main__":
    main()
