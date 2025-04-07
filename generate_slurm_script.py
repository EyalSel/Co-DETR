"""
This script generates SLURM job submission scripts for running inference on
multiple machines.
It takes a set of inference scripts (inference_part_*.sh) generated using
create_inference_scripts.py and:
1. Creates an output directory for SLURM files
2. Splits the inference tasks between two machines (manchester and ace)
3. Generates SLURM scripts for each machine with appropriate parameters
4. Saves the generated SLURM scripts to files

The script handles configuration like:
- Working directory (/data/ges/co-detr)
- Conda environment (co-detr-py310) 
- Number of GPUs per job (1)
- Output directory structure ("co-detr-inference-SLURM")
"""

from pathlib import Path
from tools.slurm_utils import get_slurm_contents

# Create directory for slurm files
output_dir = Path("co-detr-inference-SLURM")
output_dir.mkdir(exist_ok=True)

# Parameters for all jobs
cd_dir = Path("/data/ges/co-detr")
output_subdir = str(output_dir.name)
conda_env = "co-detr-py310"  # Updated conda environment name
num_gpus = 1

# Get list of all inference part scripts
inference_scripts = list(Path(".").glob("inference_part_*.sh"))

print(f"Found {len(inference_scripts)} inference scripts: {inference_scripts}")

if len(inference_scripts) == 0:
    print(
        "No inference scripts found! Make sure inference_part_*.sh files exist in the current directory."
    )
    exit(1)

# Split scripts between machines
half = len(inference_scripts) // 2
manchester_scripts = inference_scripts[:half]
ace_scripts = inference_scripts[half:]

print(f"Manchester scripts: {manchester_scripts}")
print(f"Ace scripts: {ace_scripts}")

# Generate slurm files for manchester
for i, script in enumerate(manchester_scripts):
    commands = [f"bash {script}"]
    contents = get_slurm_contents(
        machine="manchester",
        num_gpus=num_gpus,
        conda_env=conda_env,
        commands=commands,
        output_subdir=output_subdir,
        cd_dir=str(cd_dir)  # Convert Path to string for compatibility
    )

    # Save to file
    output_file = output_dir / f"slurm_manchester_{i}.sh"
    output_file.write_text(contents)

# Generate slurm files for ace
for i, script in enumerate(ace_scripts):
    commands = [f"bash {script}"]
    contents = get_slurm_contents(
        machine="ace",
        num_gpus=num_gpus,
        conda_env=conda_env,
        commands=commands,
        output_subdir=output_subdir,
        cd_dir=str(cd_dir)  # Convert Path to string for compatibility
    )

    # Save to file
    output_file = output_dir / f"slurm_ace_{i}.sh"
    output_file.write_text(contents)
