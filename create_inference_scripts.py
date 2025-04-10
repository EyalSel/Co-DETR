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
import os

from absl import app, flags
from more_itertools import distribute

from tools.copied_functions import dataset_scenarios_location

FLAGS = flags.FLAGS

# Dictionary mapping model names to their config and checkpoint paths
with open('model_registry.json', 'r') as f:
    MODEL_CONFIGS = json.load(f)

flags.DEFINE_enum('dataset',
                  default=None,
                  enum_values=[
                      "waymo",
                      "argoverse",
                      "MEVA",
                      "visdrone",
                      "kitti-step",
                  ],
                  help='Dataset to use (e.g., waymo, MEVA, visdrone)')
flags.DEFINE_enum('task',
                  default=None,
                  enum_values=[
                      'detection',
                      'instance_segmentation',
                  ],
                  help='Task type (e.g., detection, instance_segmentation)')
flags.DEFINE_multi_enum('models',
                        default=None,
                        enum_values=list(MODEL_CONFIGS.keys()),
                        help='List of models to run inference with')


# Base command
def make_cmd(model_config, scenarios_str):
    base_cmd = "bash tools/model_inference.sh {} {} 1 --dataset {} --task {} --eval bbox --scenarios {}".format(
        model_config['config'], model_config['checkpoint'], FLAGS.dataset,
        FLAGS.task, scenarios_str)
    return base_cmd


def main(_):
    # Read scenarios from JSON file
    with open(dataset_scenarios_location(FLAGS.dataset), 'r') as f:
        scenarios = json.load(f)

    scenarios_per_script = list(distribute(16, scenarios))

    # Create 16 scripts
    for i in range(16):
        script_scenarios = scenarios_per_script[i]

        # Create command with space-separated scenarios
        scenarios_str = " ".join(script_scenarios)

        # Write to shell script
        with open(f'inference_part_{i+1}.sh', 'w') as f:
            f.write("#!/bin/bash\n")
            for model_name in FLAGS.models:
                assert model_name in MODEL_CONFIGS, model_name
                assert FLAGS.task == MODEL_CONFIGS[model_name]['task'], (
                    model_name, FLAGS.task)
                full_cmd = make_cmd(MODEL_CONFIGS[model_name], scenarios_str)
                f.write(full_cmd)
                f.write("\n")

        # Make the script executable
        os.chmod(f'inference_part_{i+1}.sh', 0o755)

    print("Created 16 inference scripts!")


if __name__ == '__main__':
    app.run(main)
