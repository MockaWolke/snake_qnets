from snake_game import *
import numpy as np
import random
from tqdm import tqdm
from train_args import BasicArgs
from models import DoubleQNET
from loguru import logger

def sample_eps_greedy(greedy_action, epsilon):

    mask = np.random.uniform(0, 1, greedy_action.shape) < epsilon

    random_actions = np.random.randint(-1, 2, greedy_action.shape)

    new_actions = np.where(mask, greedy_action, random_actions)

    if new_actions.min() < -1 or new_actions.max() > 1:
        print("the fuck", new_actions)
        raise ValueError()
    return new_actions

class ReplayBuffer:

    def __init__(self, args: BasicArgs) -> None:
        self.args = args
        self.last_ops = None
        self.pos = 0
        self.size = args.buffer_size
        
        self.replays = []
        self.scores = np.zeros((self.size,), dtype=np.float32)
        
        self.priority_mode = self.args.buffer_alpha != 0.0
        
        logger.info(f"Using Prioritized Buffer - {self.priority_mode}")
        
        
    def add(self, value, priority = 1.0):
        if len(self.replays) < self.size:
            self.replays.append(value)
        else:
            self.replays[self.pos] = value
            
        self.scores[self.pos] = max(priority ** self.args.buffer_alpha, 1e-4)
        self.pos = (self.pos +1) % self.size

    def fill(self, values):
        
        for value in values:
            self.add(value)
            
    def sample(self, batchsize):
        
        if not self.priority_mode:
            
            scaled_scores = None
            
        else:
            
            scaled_scores = self.scores / self.scores.sum()

        indices = np.random.choice(len(self.replays), size=batchsize, p= scaled_scores)
        
        sample = [self.replays[idx] for idx in indices]
        
        if not self.priority_mode:
            weights = np.ones(batchsize)
        else:
        
            weights = (self.size * scaled_scores[indices]) ** (-self.args.buffer_beta)
            weights /= weights.max()

        new_obs, obs, reward, terminated, actions = map(np.array, zip(*sample))

        return new_obs, obs, reward, terminated, actions, indices, weights
    
    
    def update(self, indices, errors):
        self.scores[indices] = np.maximum(np.abs(errors) ** self.args.buffer_alpha, 1e-4)

    def init_fill(self, envs, heuristic):

        old_obs = envs.reset()[0]

        for _ in tqdm(
            range(int(np.ceil(self.size / self.args.n_envs))),
            desc="inital buffer fill",
        ):

            states = envs.get_attr("state")
            actions = np.array([heuristic(i) for i in states])
            actions = sample_eps_greedy(actions, self.args.eps)

            new_obs, reward, terminated, _, scores = envs.step(actions)

            packed = list(zip(new_obs, old_obs, reward, terminated, actions))

            self.fill(packed)

            old_obs = new_obs

        self.last_ops = new_obs
        
        assert len(self.replays) == self.size, "buffer not full"

    def fill_epoch(self, envs, model: DoubleQNET):

        n_items = int(self.size / self.args.n_epoch_refill)

        old_obs = self.last_ops

        for _ in tqdm(
            range(int(np.ceil(n_items / self.args.n_envs))), desc="Sampling new Rounds"
        ):

            actions = model.get_greedy_action(old_obs)
            actions = sample_eps_greedy(actions, self.args.eps)

            new_obs, reward, terminated, _, scores = envs.step(actions)

            packed = list(zip(new_obs, old_obs, reward, terminated, actions))
            self.fill(packed)

            old_obs = new_obs