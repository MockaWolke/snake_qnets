import yaml
import os
import argparse
import subprocess
from datetime import datetime

# Setting up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the YAML configuration file.")
parser.add_argument("--job", type=int, default=None, help="Specify a job index to run a single job.")
parser.add_argument("--p3", action="store_true")

print("Hey (:")

args = parser.parse_args()

python_path = "python3" if args.p3 else "python"

with open(args.path, "r") as f:
    config = yaml.safe_load(f)

common_settings = config["common"]
runs = config["runs"] if args.job is None else [config["runs"][args.job]]

for index, run in enumerate(runs):
    command = [python_path, "src/script.py"]
    for key, value in common_settings.items():
        command.extend([f"--{key}", str(value)])
    for key, value in run.items():
        command.extend([f"--{key}", str(value)])
    
    log_file = f"{run['run_name']}_{datetime.now().strftime('%d-%H:%M')}.log".replace(" ","_")
    with open(log_file, "w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        process.wait()
        print(f"Run {index+1} logged to {log_file}")

