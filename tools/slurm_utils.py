"""
Copied from the AV-Cloud repo.
"""

template = ("""#!/bin/bash

# the SBATCH directives must appear before any executable
# line in this script

#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=60 # number of cores per task
# I think gpu:4 will request 4 of any kind of gpu per node,
# and gpu:v100_32:8 should request 8 v100_32 per node
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --nodelist={machine} # if you need specific nodes
#SBATCH -t 5-5:00 # time requested (D-HH:MM)
# slurm will cd to this directory before running the script
# you cawn also just run sbatch submit.sh from the directory
# you want to be in
#SBATCH -D /data/ges/
# use these two lines to control the output file. Default is
# slurm-<jobid>.out. By default stdout and stderr go to the same
# place, but if you use both commands below they'll be split up
# filename patterns here: https://slurm.schedmd.com/sbatch.html
# %N is the hostname (if used, will create output(s) per node)
# %j is jobid
#SBATCH -o /home/eecs/ges/slurm_material/slurm_jobs/{output_subdir}slurm.%j.out # STDOUT
#SBATCH -e /home/eecs/ges/slurm_material/slurm_jobs/{output_subdir}slurm.%j.err # STDERR

# print some info for context
pwd
hostname
date

echo starting job...

source ~/.bashrc
conda activate {conda_env}

# python will buffer output of your script unless you set this
# if you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed
export PYTHONUNBUFFERED=1

# do ALL the research
# python train.py --config config.py
{commands}

""")


def get_slurm_contents(machine,
                       num_gpus,
                       conda_env,
                       commands,
                       output_subdir=None,
                       cd_dir=None):
    """
    This function takes the information of a SLURM job and returns the job file
    that can be directly submitted to SLURM.
     - Machine: The name of the machine used in SLURM (e.g. manchester)
     - num_gpus: The number of GPUs to use on the machine
     - conda_env: The conda environment to run the commands on
     - commands: A list of strings specifying the commands that will be run
     - output_subdir: The name of the subdirectory in which the job out and err
       files are logged
     - cd_dir: The working directory in which the commands will be executed
    """
    assert len(commands) > 0
    if output_subdir is None:
        output_subdir = ""
    else:
        output_subdir = output_subdir + "/"
    if cd_dir is not None:
        commands = ["cd " + cd_dir + " && " + cmd for cmd in commands]
    commands = "\n".join(commands)
    return template.format(machine=machine,
                           num_gpus=num_gpus,
                           conda_env=conda_env,
                           commands=commands,
                           output_subdir=output_subdir)
