import subprocess
import time
import argparse
import yaml

def run_simulation_subprocess(l_sticky_value, alpha_sticky_deg_value, n_steps, save_every, plot_every, random_placement, random_chance, batch_mode=True):
    # Build the command to run your simulation script with arguments
    command = [
        'python', 'simulation_assembly.py',  # Replace with your actual script name
        '--l_sticky', str(l_sticky_value),
        '--alpha_sticky_deg', str(alpha_sticky_deg_value),
        '--n_steps', str(n_steps),
        '--save_every', str(save_every),
        '--plot_every', str(plot_every),
        '--random_chance', str(random_chance)
    ]

    if random_placement:
        command.append('--random_placement')

    if batch_mode:
        command.append('--batch_mode')

    process = subprocess.Popen(command)
    return process

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run multiple simulations based on configuration file.')
    parser.add_argument('--config_file', type=str, default='simulation_config.yaml', help='Path to the configuration file')
    parser.add_argument('--n_steps', type=int, default=10000000, help='Number of steps for each simulation (default: 10000000)')
    parser.add_argument('--max_concurrent', type=int, default=18, help='Maximum number of concurrent simulations (default: 4)')
    parser.add_argument('--save_every', type=int, default=10000, help='Frequency to save simulation (default: 5000)')
    parser.add_argument('--plot_every', type=int, default=5000, help='Frequency to plot simulation (default: 5000)')
    args = parser.parse_args()

    config_file = args.config_file
    n_steps = args.n_steps
    max_concurrent_processes = args.max_concurrent
    save_every = args.save_every
    plot_every = args.plot_every

    # Read the configuration file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    simulations = config.get('simulations', [])

    processes = []
    for sim_entry in simulations:
        l_sticky = sim_entry.get('l_sticky', 3.0)
        random_placement = sim_entry.get('random_placement', False)
        random_chance = sim_entry.get('random_chance', 0.0)
        repeats = sim_entry.get('repeats', 1)
        alpha_values = sim_entry.get('alpha_sticky_deg_values', [])

        for alpha_sticky_deg in alpha_values:
            for i in range(repeats):
                # Start the subprocess
                process = run_simulation_subprocess(
                    l_sticky,
                    alpha_sticky_deg,
                    n_steps,
                    save_every,
                    plot_every,
                    random_placement,
                    random_chance,
                    batch_mode=True
                )
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
