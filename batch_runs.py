import numpy as np
import subprocess
import time
import argparse

def run_simulation_subprocess(delta_value, n_steps, add_unit_every, save_every, plot_every):
    # Build the command to run your simulation script with arguments
    command = [
        'python', 'simulation_assembly.py',  # Replace with the actual script name
        '--delta', str(delta_value),
        '--n_steps', str(n_steps),
        '--add_unit_every', str(add_unit_every),
        '--save_every', str(save_every),
        '--plot_every', str(plot_every)
    ]
    process = subprocess.Popen(command)
    return process

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run multiple simulations with varying DELTA values.')
    parser.add_argument('--delta_start', type=float, default=0.1, help='Starting value of DELTA range (default: 0.2)')
    parser.add_argument('--delta_end', type=float, default=0.01, help='Ending value of DELTA range (default: 0.01)')
    parser.add_argument('--num_simulations', type=int, default=16, help='Total number of simulations to run (default: 32)')
    parser.add_argument('--n_steps', type=int, default=100000000, help='Number of steps for each simulation (default: 100000000)')
    parser.add_argument('--max_concurrent', type=int, default=16, help='Maximum number of concurrent simulations (default: 16)')
    parser.add_argument('--add_unit_every', type=int, default=500, help='Frequency to add unit')
    parser.add_argument('--save_every', type=int, default=5000, help='Frequency to save simulation')
    parser.add_argument('--plot_every', type=int, default=10000, help='Frequency to plot simulation')
    args = parser.parse_args()

    delta_start = args.delta_start
    delta_end = args.delta_end
    num_simulations = args.num_simulations
    n_steps = args.n_steps
    max_concurrent_processes = args.max_concurrent
    add_unit_every = args.add_unit_every
    save_every = args.save_every
    plot_every = args.plot_every

    # Generate DELTA values, larger DELTA values first
    delta_values = np.linspace(delta_start, delta_end, num_simulations)
    delta_values = sorted(delta_values, reverse=True)  # Ensure largest DELTA values run first

    processes = []
    for delta_value in delta_values:
        # Start the subprocess
        process = run_simulation_subprocess(delta_value, n_steps, add_unit_every, save_every, plot_every)
        processes.append(process)

        # Check if we've reached the maximum number of concurrent processes
        while len(processes) >= max_concurrent_processes:
            # Poll processes to see if any have finished
            for p in processes:
                if p.poll() is not None:
                    # Process has finished
                    processes.remove(p)
                    break
            else:
                # No processes have finished yet; wait a bit before checking again
                time.sleep(1)

    # Wait for any remaining processes to finish
    for p in processes:
        p.wait()
