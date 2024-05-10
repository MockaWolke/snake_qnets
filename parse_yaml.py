import yaml
import os
import argparse
import subprocess
from datetime import datetime
from src.train_args import BasicArgs
import matplotlib.pyplot as plt
import numpy as np

types =  {a:str(b.annotation) for a,b in BasicArgs.model_fields.items()}

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the YAML configuration file.")
parser.add_argument("--job", type=int, default=None, help="Specify a job index to run a single job.")
parser.add_argument("--p3", action="store_true")
parser.add_argument("--o", type=str, default="runs.sh")
parser.add_argument("--plot", action="store_true")

args = parser.parse_args()

python_path = "python3" if args.p3 else "python"

with open(args.path, "r") as f:
    config = yaml.safe_load(f)

common_settings = config["common"]
runs = config["runs"] if args.job is None else [config["runs"][args.job]]


def write(common_settings, runs):
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
    
        log_file = f"{run['run_name']}.log".replace(" ","_")
    # command.extend([">", log_file, "2>&1"])

        commands.append(" ".join(command))


    with open(args.o , "w") as f:
        f.write("\n".join(commands))
        

def plot_examples(common_settings : dict, runs):
    
    fix, ax = plt.subplots(3,1, figsize = (6,6))
    
    
    for run in runs:
        
        se = common_settings
        se.update(run)
        
        steps = se["steps_per_epoch"] * se["epochs"]
        
        sett = BasicArgs(**se)
        
        if sett.eps_anneal_rate == 1.0: # linear
            
            ax[0].plot([0,sett.eps_anneal_steps, steps], [sett.eps, sett.min_eps, sett.min_eps], label = se["run_name"])
            
        else: 
            
            eps = np.maximum(sett.eps * (sett.eps_anneal_rate ** np.arange(steps)), sett.min_eps)
            ax[0].plot(eps, label = se["run_name"])
            
        # lr
        
        lr_anneal_steps = min(sett.lr_anneal_steps or steps, steps)
        
        
        min_lr = se.get("min_lr", sett.lr)
        ax[1].plot([0,lr_anneal_steps, steps], [sett.lr,min_lr, min_lr], label = se["run_name"])
        
        beta = np.minimum(sett.buffer_beta + sett.incr_buffbeta * np.arange(steps), 1)
        ax[2].plot(beta, label = se["run_name"])
        
    
    ax[0].set_title("Eps")
    ax[1].set_title("Lr")
    ax[2].set_title("Buffer Beta")
    ax[0].legend(loc = "best")
    plt.tight_layout()
    plt.show()


if args.plot:
    plot_examples(common_settings, runs)
else:
    write(common_settings, runs)
    

