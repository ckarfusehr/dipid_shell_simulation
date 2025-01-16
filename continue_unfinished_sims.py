import os
import pickle
import argparse
import numpy as np
from pathlib import Path
from simulation_assembly import MolecularDynamicsSimulation, SimulationVisualizer, run_simulation

# Directory paths
UNFINISHED_DIR = Path("unfinished_sims")
PROCESSED_DIR = UNFINISHED_DIR / "processed"

# Ensure processed directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_simulation(file_path):
    """Load a simulation object from a file."""
    with open(file_path, 'rb') as f:
        sim = pickle.load(f)
    return sim

def save_simulation(sim, file_path):
    """Save a simulation object to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(sim, f)

def continue_simulation(sim, n_steps, add_unit_every, save_every_batch, plot_every_batch, equilibrium_threshold, equilibrium_threshold_absolute, batch_mode, scaling, plot_outer_layer):
    """Continue the simulation from its current state."""
    visualizer = None
    if not batch_mode:
        visualizer = SimulationVisualizer(sim, scaling=scaling, plot_outer_layer=plot_outer_layer)

    run_simulation(
        sim, 
        visualizer, 
        n_steps, 
        add_unit_every, 
        save_every_batch, 
        plot_every_batch, 
        'simulation',
        equilibrium_threshold=equilibrium_threshold, 
        equilibrium_threshold_absolute=equilibrium_threshold_absolute
    )

def process_single_simulation(args):
    """Process a single unfinished simulation passed via arguments."""
    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Specified file {file_path} does not exist.")
        return

    try:
        print(f"Processing simulation for: {file_path.name}")

        # Load the simulation
        sim = load_simulation(file_path)

        # Continue the simulation
        continue_simulation(
            sim,
            n_steps=args.n_steps,
            add_unit_every=args.add_unit_every,
            save_every_batch=args.save_every_batch,
            plot_every_batch=args.plot_every_batch,
            equilibrium_threshold=1e-6,
            equilibrium_threshold_absolute=1e-15,
            batch_mode=args.batch_mode,
            scaling=sim.monomer_info['scaling'],
            plot_outer_layer=True
        )

        # Save the processed simulation
        processed_file = PROCESSED_DIR / file_path.name
        save_simulation(sim, processed_file)

        # Remove the original file
        file_path.unlink()

        print(f"Simulation processed and saved: {processed_file.name}")
    except Exception as e:
        print(f"Error processing simulation {file_path.name}: {e}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Continue a single unfinished molecular dynamics simulation.")
    parser.add_argument('--file', type=str, required=True, help="Path to the specific unfinished simulation file to process.")
    parser.add_argument('--save_every_batch', type=int, default=1, help="Steps interval to save simulation state (default: 1)")
    parser.add_argument('--plot_every_batch', type=int, default=1, help="Steps interval to plot simulation state (default: 1)")
    parser.add_argument('--n_steps', type=int, default=1000000, help="Total number of simulation steps (default: 1000000)")
    parser.add_argument('--batch_mode', action='store_true', help="Run simulation in batch mode without plotting")
    parser.add_argument('--add_unit_every', type=int, default=20000, help="Steps interval to add new units (default: 40000)")

    args = parser.parse_args()

    process_single_simulation(args)
