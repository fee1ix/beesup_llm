import os
import sys
import yaml
import subprocess

import logging
def set_info():
    logger = logging.getLogger('beesup_llm')
    logger.setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

def set_debug():
    logger = logging.getLogger('beesup_llm')
    logger.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

# Access the input arguments
if len(sys.argv) < 2:
    print("Usage: python script.py <input_argument>")
    sys.exit(1)

multirun_config_path = sys.argv[1]
with open(multirun_config_path, 'r') as file:
    multirun_config = yaml.safe_load(file)

for framework_dir in multirun_config['framework_dirs']:
    sys.path.append(framework_dir)

from beesup_llm.toolkit.setup_utils import *

for i, experiment_dir in enumerate(multirun_config['experiment_dirs']):
    experiment_config=load_yaml(f"{experiment_dir}/config.yaml")

    print(f"RUNNING EXPERIMENT {i+1}/{len(multirun_config['experiment_dirs'])}\t{experiment_config['name']}")

    set_debug()

    stdout_log = os.path.join(experiment_dir, f"stdout.log")
    stderr_log = os.path.join(experiment_dir, f"stderr.log")

    with open(stdout_log, "w") as stdout_file, open(stderr_log, "w") as stderr_file:
        process = subprocess.Popen(
            #f"conda run -n beesup python {extraction_experiment.__file__} {experiment_path}",
            #shell=True,
            ["python", multirun_config['module_path'], experiment_dir],
            stdout=stdout_file,
            stderr=stderr_file,
            env={
                **os.environ,
                'PYTHONPATH': multirun_config['framework_dirs'][0],
            },
        )
        process.wait()  # Wait for the experiment to finish


    if process.returncode == 0:
        print(f"EXPERIMENT {i+1}/{len(multirun_config['experiment_dirs'])}\t{experiment_config['name']} completed successfully.")
    else:
        print(f"EXPERIMENT {i+1}/{len(multirun_config['experiment_dirs'])}\t{experiment_config['name']} failed.")




