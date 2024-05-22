from snake_game import *
import numpy as np
import torch
from tqdm import tqdm
from train_args import BasicArgs
import torch.nn as nn
from functools import lru_cache

from env_utils import EnvApp, pred_to_img, preprocess_input, preprocess_label, cast_to_tensor


class MarkovSampler(nn.Module):

    def __init__(
        self,
        agent,
        env_model: EnvApp,
        device: str,
        agent_args: BasicArgs,
        max_depth: int = 10,
    ) -> None:
        super().__init__()

        self.agent = agent.to(device=device)
        self.env_model = env_model.to(device=device)
        self.device = device
        self.max_depth = max_depth

        self.colors = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        self.agent_args = agent_args

    def greedy_agent_action(self, obs: torch.Tensor) -> np.ndarray:

        # get last frame
        obs = obs[:, -1]
        
        obs = cast_to_tensor(obs, self.device, torch.float32).permute(
            0, 3, 1, 2
        )

        with torch.no_grad():

            return torch.argmax(self.agent(obs), -1) - 1

    def get_q_values(self, obs):
        
        if obs.shape[1] == 3:
            obs = obs[:, -1]
            
        obs = obs.to(dtype=torch.float32, device=self.device).permute(
            0, 3, 1, 2
        )

        with torch.no_grad():

            return self.agent(obs)

    def pred_env(
        self, obs: torch.Tensor, actions: torch.Tensor, with_raw=False
    ) -> torch.Tensor:

        x = preprocess_input(obs, actions, self.device)

        with torch.no_grad():
            pred = torch.argmax(self.env_model(x), dim=1)

        img, rewards = pred_to_img(pred, self.colors, device=self.device)

        if with_raw:
            return img.cpu(), rewards.cpu(), pred

        return img.cpu(), rewards.cpu()

    def gen_worst_next_state(self, img: np.array, max_diff=60):
        
        if torch.is_tensor(img):
            img = img.cpu().numpy()

        black = np.array([0, 0, 0])[None, None, :]
        posible_positions = np.argwhere((img == black).all(-1))

        n_apples = min(max_diff, len(posible_positions))
        apples = np.random.choice(len(posible_positions), n_apples, replace=False)

        new_images = np.stack([img for _ in range(n_apples)])
        x, y = posible_positions[apples][:, 0], posible_positions[apples][:, 1]

        new_images[np.arange(n_apples), x, y, 1] = 1


        q_vals = self.get_q_values(torch.tensor(new_images)).max(-1).values.cpu().numpy()
        worst_id = np.argmin(q_vals)
        return new_images[worst_id]
    
    def choose_action(self, obs):
        
        actions = [-1, 0, 1]
        
        obs = torch.tensor(obs)
        def key(action):
            
            state = self.gen_next_states(obs, (action,))
            
            return self.recu_search(state)
        
        return max( actions, key=key)
            
            
    def greedy_search(self, obs):
        
        obs = cast_to_tensor(obs, self.device, torch.float32)
        
        actions = cast_to_tensor((-1, 0, 1), self.device, torch.long)
        
        states, scores = self.gen_next_states(obs, actions)

        terminated = scores == -1

        for _ in range(self.max_depth):
            
            if terminated.all():
                break
            
            actions = self.greedy_agent_action(states)
            states, reward = self.gen_next_states(states, actions)
            
            terminated = (reward == -1) | terminated
            
            scores += (1 - terminated.to(torch.float32)) * reward
            
            
        return actions[torch.argmax(scores)]


    def gen_next_states(self, obs : torch.Tensor, actions = [-1, 0, 1]):

        if len(obs.shape) == 4:

            obs = torch.stack([obs for _ in actions])

        next_states, rewards = self.pred_env(obs, actions)
        
        for i, r in enumerate(rewards):

            new_state = torch.tensor(self.gen_worst_next_state(next_states[i]))
            

            if r == 1:
                
                next_states[i] = new_state
                
        next_states = torch.cat((obs[:,1:], next_states.unsqueeze(1)), dim = 1)
        


        return next_states, rewards

    def recu_search(self, obs, depth=0):
        
        obs = cast_to_tensor(obs, "cpu", torch.float32)

        if depth == self.max_depth:
            return self.get_q_values(obs[None, :]).max()
            

        next_states, rewards = self.gen_next_states(obs)

        val = max(
            self.recu_search(state, depth= depth + 1) + reward if reward != -1 else -1
            for state, reward in zip(next_states, rewards)
        )

        return val

    def eval_env_model_performance(self, n_steps):

        def com_label_acc(img, obs, reward):

            label = preprocess_label(obs[-1].copy(), reward)

            return (img[0].cpu().numpy() == label).all()

        env = SnakeGame(
            width=self.agent_args.width_and_height,
            height=self.agent_args.width_and_height,
            border=self.agent_args.border,
            food_amount=1,
            render_mode="human",
            manhatten_fac=0,
            mode="eval",
        )

        env = FrameStack(env)

        ar_acc = 0
        label_acc = 0
        ar_counter = 0
        reward_acc = 0
        scores = []
        score = 0

        obs = env.reset()[0]

        self.env_model.eval()

        for _ in tqdm(range(n_steps), desc="Evaluating Env Model performance"):

            action = self.greedy_agent_action(obs[None, :].copy()).cpu().numpy()
            
         
                
            ar, re, pred = self.pred_env(obs[None, :].copy(), action, with_raw=True)

            obs, reward, done, truncated, info = env.step(action[0])
            
            score += reward
        
        
            reward_acc += re[0] == reward
            if reward == 0:
                ar_acc += np.all(ar.numpy()[0] == obs[-1])
                ar_counter += 1
                label_acc += com_label_acc(torch.clone(pred), obs.copy(), reward)
                

            if done:
                scores.append(score)
                score = 0
                obs = env.reset()[0]

        if not done:
            scores.append(score)
            


        return ar_acc / ar_counter, label_acc / ar_counter, reward_acc / n_steps, np.mean(scores)
