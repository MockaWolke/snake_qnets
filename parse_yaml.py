import yaml
import os
import argparse
import subprocess
from datetime import datetime
from src.train_args import BasicArgs

types =  {a:str(b.annotation) for a,b in BasicArgs.model_fields.items()}

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the YAML configuration file.")
parser.add_argument("--job", type=int, default=None, help="Specify a job index to run a single job.")
parser.add_argument("--p3", action="store_true")
parser.add_argument("--o", type=str, default="runs.sh")



args = parser.parse_args()

python_path = "python3" if args.p3 else "python"

with open(args.path, "r") as f:
    config = yaml.safe_load(f)

common_settings = config["common"]
runs = config["runs"] if args.job is None else [config["runs"][args.job]]


commands = []

for index, run in enumerate(runs):
    command = [python_path, "src/script.py"]
    for key, value in common_settings.items():
        
        s_val = str(value)
        if types[key] in [str(type("a")), 'typing.Optional[str]']:
            s_val = '"' + s_val + '"'
        
        command.extend([f"--{key}", str(s_val)])
    
    for key, value in run.items():
        s_val = str(value)
        if types[key] in [str(type("a")), 'typing.Optional[str]']:
            s_val = '"' + s_val + '"'
        command.extend([f"--{key}", str(s_val)])
    
    log_file = f"{run['run_name']}_{datetime.now().strftime('%d-%H:%M')}.log".replace(" ","_")
    command.extend([">", log_file, "2>&1"])

    commands.append(" ".join(command))


with open(args.o , "w") as f:
    f.write("\n".join(commands))

