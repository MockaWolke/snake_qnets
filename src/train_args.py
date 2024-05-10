from typing import Optional, List, Tuple, Set, Dict
import sys
from pydantic import BaseModel
from datetime import datetime
import torch
from enum import Enum

class UpdateModes(str, Enum):
    epoch_copy = 'epoch'
    polyak = 'polyak'

class BasicArgs(BaseModel):
    
    #misc
    device: Optional[str] = None
    test : bool = False
    
    # envs
    n_envs : int = 8
    width_and_height : int = 14
    food_amount : int = 1
    border : int = 1
    seed : int = 0
    manhatten_fac : float = 0.05
    max_eval_steps : int = 10000
    eval_envs : int = 16
    
    #wandb
    run_name : str = "unnamed"
    wandb_mode : str = "offline"
    use_wandb : bool = True
    run_group : Optional[str] = None
    make_vid : bool = True
    
    #model
    backbone : str = "normal"
    update_mode : UpdateModes = UpdateModes.epoch_copy
    
    
    #hpams
    buffer_size : int = 100_000
    batch_size : int = 32
    update_freq : int = 1
    
    eps : float = 0.1
    min_eps : float = 0.05
    eps_anneal_steps : int = 50000
    eps_anneal_rate : float  = 1.0
    gamma : float = 0.98
    lr : float = 1e-3
    epochs : int = 50
    steps_per_epoch : int = 50
    init_step : int = 0
    n_epoch_refill : int = 10
    eval_freq : int = 2
    min_lr : Optional[float] = None
    theta : float = 0.005
    buffer_alpha : float = 0.0
    buffer_beta : float = 0.0
    incr_buffbeta : float = 0.0
    lr_anneal_steps : Optional[int] = None
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        if self.test:
            self.run_name = self.run_name + "_test"
            self.epochs = 2
            self.steps_per_epoch = 2
            self.use_wandb = False
        
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print("setting device to", self.device)
            
        self.run_name = self.run_name + "_" + datetime.now().strftime('%d-%m-%H:%M:%S') 
        
        if self.min_lr is not None:
            print(f"Annealing learning rate to {self.min_lr}")
            
        if self.eps_anneal_rate == 1.0:
            print("linear eps anneal")
        else:
            print("exponential eps decay")
        
            
            
def parse_args(Cls):
    """Parse cli args to model"""
    
    if len(sys.argv) > 1:
        
        if sys.argv[1] in ["help", "--help"]:
            
            
            print("the possible keys are:")
            
            
            for name, field in Cls.model_fields.items():
                print(f"    {name} - {field.annotation}")
            
            sys.exit()
            
        
        names = sys.argv[1::2]
        vals = sys.argv[2::2]
        
        if len(names) != len(vals):
            raise ValueError("uneven args")
    
        possible_keys = set(Cls.model_fields.keys())
    
    
        for i,n in enumerate(names):
            
            n = n.replace("--", "")
            if n not in possible_keys:
                raise ValueError(f"{n} not found in model")
    
            names[i] = n
    

        
        kwargs = dict(zip(names, vals))
    else:
        kwargs = {}
        
    return Cls(**kwargs)

