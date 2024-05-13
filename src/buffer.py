from snake_game import *
import numpy as np
import random
from tqdm import tqdm
from train_args import BasicArgs
from models import DoubleQNET
from loguru import logger

def sample_eps_greedy(greedy_action, epsilon):

    mask = np.random.uniform(0, 1, greedy_action.shape) > epsilon

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

        # init env
        self.last_ops = envs.reset()[0]

        def func(env, obs):
            states = envs.get_attr("state")
            actions = np.array([heuristic(i) for i in states])
            return actions
        
        
        self._sample_env(int(np.ceil(self.size / self.args.n_envs)), envs, func, "Initial Fill")
        
        assert len(self.replays) == self.size, "buffer not full"
        
        
    def _sample_env(self, steps: int, envs, sample_func, desc = None):
        
        old_obs = self.last_ops

        samples = []

        for _ in tqdm(
            range(steps + self.args.n_obs_reward - 1), desc=desc
        ):

            action = sample_func(envs, old_obs)

            actions = sample_eps_greedy(action, self.args.eps)

            new_obs, reward, terminated, _, scores = envs.step(actions)

            samples.append((new_obs, old_obs, reward, terminated, actions))

            old_obs = new_obs
            
        # important set last ops 
        self.last_ops = new_obs
            
        
        for idx in range(len(samples) - self.args.n_obs_reward + 1): # cancels out with 1
            
            
            _, old_obs, _, _, actions = samples[idx] 
            
            
            s_reward, s_terminated = [], []
            
            
            for i in range(self.args.n_obs_reward):
                
                new_obs, _, reward, terminated, _ = samples[idx + i] # get respective future points
                
                s_reward.append(reward)
                s_terminated.append(terminated)

            
            s_reward, s_terminated = map(lambda x: np.stack(x, axis=1),(s_reward, s_terminated))
            
            assert s_reward.shape == (self.args.n_envs, self.args.n_obs_reward)
            
            # unpack over enironments
            packed = list(zip(new_obs, old_obs, s_reward, s_terminated, actions))
            self.fill(packed)
        

    def fill_epoch(self, envs, model: DoubleQNET):

        n_items = int(self.size / self.args.n_epoch_refill)

        steps = int(np.ceil(n_items / self.args.n_envs))
        
        def func(env, obs):
            return model.get_greedy_action(obs)
        
        self._sample_env(steps, envs, func, desc = "Samling new Rounds")