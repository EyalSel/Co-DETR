"""
This script creates 16 separate shell scripts for running inference on Waymo
scenarios.
It:
1. Reads scenarios from a hardcoded JSON file (waymo_scenarios.json)
2. Splits them evenly across 16 scripts (for parallel processing)
3. Creates shell scripts with appropriate commands and scenarios
4. Makes the scripts executable

The generated scripts will be named inference_part_1.sh through
inference_part_16.sh.
Each script runs the waymo_inference.sh script with a subset of scenarios.

These scripts are then packaged for submission to the SLURM scheduler using
generate_slurm_script.py.
"""
import json
import math

# Read scenarios from JSON file
with open('waymo_scenarios.json', 'r') as f:
    scenarios = json.load(f)

# Calculate how many scenarios per script (round up to ensure all scenarios are covered)
scenarios_per_script = math.ceil(len(scenarios) / 16)

# Base command
base_cmd = "bash tools/waymo_inference.sh projects/configs/co_dino/co_dino_5scale_vit_large_coco.py co_detr_vit/co_dino_5scale_vit_large_coco.pth 1 --eval bbox"

# Create 16 scripts
for i in range(16):
    start_idx = i * scenarios_per_script
    end_idx = min((i + 1) * scenarios_per_script, len(scenarios))
    
    # Get subset of scenarios for this script
    script_scenarios = scenarios[start_idx:end_idx]
    
    # Create command with space-separated scenarios
    scenarios_str = " ".join(script_scenarios)
    full_cmd = f"{base_cmd} --scenarios {scenarios_str}"
    
    # Write to shell script
    with open(f'inference_part_{i+1}.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(full_cmd)
    
    # Make the script executable
    import os
    os.chmod(f'inference_part_{i+1}.sh', 0o755)

print("Created 16 inference scripts!") 