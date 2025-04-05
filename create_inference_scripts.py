"""
This script creates 16 separate shell scripts for running inference on the
scenarios of a dataset.

It:
1. Reads scenarios from a hardcoded JSON file (e.g. waymo_scenarios.json)
2. Splits them evenly across 16 scripts (for parallel processing)
3. Creates shell scripts with appropriate commands and scenarios
4. Makes the scripts executable

The generated scripts will be named inference_part_1.sh through
inference_part_16.sh.
Each script runs the model_inference.sh script with a subset of scenarios.

These scripts are then packaged for submission to the SLURM scheduler using
generate_slurm_script.py.
"""
import json
import math
from tools.copied_functions import dataset_scenarios_location

# dataset = "waymo"
# dataset = "MEVA"
dataset = "visdrone"

# Read scenarios from JSON file
with open(dataset_scenarios_location(dataset), 'r') as f:
    scenarios = json.load(f)

# Dictionary mapping model names to their config and checkpoint paths
with open('model_registry.json', 'r') as f:
    MODEL_CONFIGS = json.load(f)

# Calculate how many scenarios per script (round up to ensure all scenarios are covered)
scenarios_per_script = math.ceil(len(scenarios) / 16)

# Base command
def make_cmd(model_config, scenarios_str):
    base_cmd = "bash tools/model_inference.sh {} {} 1 --dataset {} --eval bbox --scenarios {}".format(
        model_config['config'],
        model_config['checkpoint'],
        dataset,
        scenarios_str
    )
    return base_cmd

# Create 16 scripts
for i in range(16):
    start_idx = i * scenarios_per_script
    end_idx = min((i + 1) * scenarios_per_script, len(scenarios))

    # Get subset of scenarios for this script
    script_scenarios = scenarios[start_idx:end_idx]

    # Create command with space-separated scenarios
    scenarios_str = " ".join(script_scenarios)
    
    # Write to shell script
    with open(f'inference_part_{i+1}.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        for model_name, model_config in MODEL_CONFIGS.items():
            full_cmd = make_cmd(model_config, scenarios_str)
            f.write(full_cmd)
            f.write("\n")

    # Make the script executable
    import os
    os.chmod(f'inference_part_{i+1}.sh', 0o755)

print("Created 16 inference scripts!")
