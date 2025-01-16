import subprocess
import time
import argparse
from pathlib import Path

def run_simulation_subprocess(file_path, n_steps, save_every_batch, plot_every_batch, add_unit_every, batch_mode=True):
    """Run a single simulation subprocess."""
    command = [
        'python', 'continue_unfinished_sims.py',
        '--file', str(file_path),
        '--n_steps', str(n_steps),
        '--save_every_batch', str(save_every_batch),
        '--plot_every_batch', str(plot_every_batch),
        '--add_unit_every', str(add_unit_every),
    ]

    if batch_mode:
        command.append('--batch_mode')

    process = subprocess.Popen(command)
    return process

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run batch processing for multiple unfinished simulations.')
    parser.add_argument('--n_steps', type=int, default=1000000, help='Total number of simulation steps (default: 1000000)')
    parser.add_argument('--max_concurrent', type=int, default=5, help='Maximum number of concurrent simulations (default: 5)')
    parser.add_argument('--save_every_batch', type=int, default=50, help='Frequency to save simulation state (default: 50)')
    parser.add_argument('--plot_every_batch', type=int, default=200, help='Frequency to plot simulation state (default: 200)')
    parser.add_argument('--add_unit_every', type=int, default=20000, help='Steps interval to add new units (default: 40000)')

    args = parser.parse_args()

    unfinished_dir = Path("unfinished_sims")
    unfinished_files = list(unfinished_dir.glob("*.pkl"))

    if not unfinished_files:
        print("No unfinished simulations found.")
        exit()

    max_concurrent_processes = args.max_concurrent
    processes = []

    while unfinished_files or processes:
        # Start new processes if below max_concurrent
        while len(processes) < max_concurrent_processes and unfinished_files:
            file = unfinished_files.pop(0)
            try:
                print(f"Starting simulation for: {file.name}")

                process = run_simulation_subprocess(
                    file,
                    args.n_steps,
                    args.save_every_batch,
                    args.plot_every_batch,
                    args.add_unit_every,
                    batch_mode=True
                )
                processes.append((process, file))
            except Exception as e:
                print(f"Error starting simulation for {file.name}: {e}")

        # Check running processes
        for process, file in processes[:]:
            if process.poll() is not None:  # Process finished
                processes.remove((process, file))
                print(f"Simulation finished for: {file.name}")

        # Wait before checking again
        time.sleep(10)

    print("Batch processing complete.")
