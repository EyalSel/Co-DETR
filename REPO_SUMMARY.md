# Repo summary

## General
1. Environment setup: environment_setup.sh
2. (hardcoded) Dataset scenarios: dataset_scenarios/*_scenarios.json
3. Model config: model_registry.json
4. Copied over functions: tools/copied_functions.py

## Inference
1. Main script: tools/model_inference.py (see create_inference_scripts.py for
   usage details)
2. Create inference scripts for dataset: create_inference_scripts.py
3. Generate SLURM scripts from inference scripts: generate_slurm_script.py

## Latency profiling
1. Main script: profile_codetr_latency.py
2. Sweep script: sweep_profile_results.sh
