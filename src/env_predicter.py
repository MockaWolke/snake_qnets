from snake_game import *
import numpy as np
import gymnasium as gym
import torch
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from typing import Optional, List, Tuple, Set, Dict
from train_args import BasicArgs, parse_args
from models import DoubleQNET
from collections import deque
import wandb
import shutil
from utils import make_vid
from loguru import logger
from buffer import ReplayBuffer, sample_eps_greedy
from torch.utils.data import Dataset, DataLoader
from lightning import LightningModule, Trainer
from pydantic import BaseModel
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, MeanMetric
import gdown
import yaml
import sys
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from models import Backbone

class NewArgs(BaseModel):
    data_points: int = 10
    n_steps_fut: int = 3
    batchsize: int = 64
    device: str = "cpu"
    m_width: int = 64
    lr: float = 1e-3
    test: bool = True
    epochs: int = 32
    n_workers: int = 8
    run_name: str
    wandb_mode: str = "online"
    loss_weight : float = 10.0
    val_steps : int = 1000


def gen_data(size, eps, agent, envs, agent_args, args):
    data = []

    old_obs = envs.reset()[0]

    samples = []

    steps = int(np.ceil(size / agent_args.n_envs))

    unique_vals = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )

    def sample_func(env, obs):
        return agent.get_greedy_action(obs)

    for _ in tqdm(
        range(steps + agent_args.n_obs_reward - 1), desc=f"sample for eps {eps}"
    ):

        action = sample_func(envs, old_obs)

        actions = sample_eps_greedy(action, eps)

        new_obs, reward, terminated, _, scores = envs.step(actions)

        samples.append((new_obs, old_obs, reward, terminated, actions))

        old_obs = new_obs

    for idx in range(len(samples) - agent_args.n_obs_reward + 1):


        s_reward, s_terminated, old_obs = [], [], []

        for i in range(agent_args.n_obs_reward - 1):

            new_obs, old, reward, terminated, actions = samples[
                idx + i
            ]  

            s_reward.append(reward)
            s_terminated.append(terminated)
            old_obs.append(old)

        s_reward, s_terminated, old_obs = map(
            lambda x: np.stack(x, axis=1), (s_reward, s_terminated, old_obs)
        )

        for tup in zip(old_obs, s_reward, s_terminated, actions, new_obs):
            old_obs, s_reward, s_terminated, actions, new_obs = tup
            old_obs[:, [0, -1]] = 0
            old_obs[:, :, [0, -1]] = 0
            label = new_obs
            label[:, [0, -1]] = 0
            label[[0, -1], :] = 0
            reward = s_reward[-1]

            if s_terminated[:-1].any():
                continue

            label = (
                (label[:, :, None, :] == unique_vals[None, None, :])
                .all(-1)
                .astype(np.int32)
            )

            if reward == 1:
                label[label[:, :, 1] == 1, 0] = 1
                label[label[:, :, 1] == 1, 1] = 0

            elif reward == -1:
                label[:, :, 0] = 1
                label[:, :, 1:] = 0

            diffs = (old_obs[:-1] != old_obs[1:]).any(-1).sum((1, 2))
            if not np.isin(diffs, [2, 3]).all():
                raise ValueError(f"logic fucked {diffs}")

            old_obs = np.concatenate(old_obs, -1)
            action_mask = np.zeros((16, 16, 3))
            action_mask[:, :, actions + 1] = 1
            x = np.concatenate((old_obs, action_mask), axis=-1)
            y = label

            data.append((x, y))

        packed = list()
        data.extend(packed)

    return data


class EnvDataset(Dataset):

    def __init__(self, data) -> None:
        super().__init__()

        self.data = data

    def __getitem__(self, index):

        x, y = self.data[index]
        x = torch.tensor(x, dtype=torch.float32).permute((2, 0, 1))

        y = np.argmax(y, -1)

        y = torch.tensor(y, dtype=torch.long)

        return x, y

    def __len__(self):
        return len(self.data)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.use_skip = in_channels == out_channels

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) 
        if self.use_skip:
            x = x + identity
        return x



class EnvApp(Module):

    def __init__(self, args: NewArgs) -> None:

        super().__init__()

        w = args.m_width

        self.block1 = ConvBlock((args.n_steps_fut + 1) * 3, w)
        self.block2 = ConvBlock(w , w * 2)


        self.pool = nn.MaxPool2d(2)
        self.smaller_block1 = ConvBlock(w * 2, w* 4)
        self.smaller_block2 = ConvBlock(w * 4, w * 2)
        
        self.inv = nn.ConvTranspose2d(
            w * 2, w * 2, 3, padding=1, output_padding=1, stride=2
        )
        
        self.block3 = ConvBlock(w * 4, w * 2)
        self.block4 = ConvBlock(w * 2, w * 2)
        
        self.out_layer = nn.Conv2d(w * 2, 4, 1, padding=0)

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        
        smaller = self.pool(x)
        smaller = self.smaller_block1(smaller)
        smaller = self.smaller_block2(smaller)
        
        x = torch.cat([x, self.inv(smaller)], dim = 1)
        
        x = self.block3(x) 
        x = self.block4(x) 

        return self.out_layer(x)

class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack=3):
        super(FrameStack, self).__init__(env)
        self.num_stack = num_stack
        self.frames = []
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(num_stack, env.height + env.border * 2, env.width + env.border * 2, 3), 
            dtype=np.float32
        )
    
    def reset(self):
        obs,info = self.env.reset()
        self.frames = [obs for _ in range(self.num_stack)]
        return self._get_observation(), info
    
    def step(self, action):
        
        obs, reward, done, truncated, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_observation(), reward, done, truncated, info
    
    def _get_observation(self):
        return np.stack(self.frames, axis=0)
    
class MarkovSampler(nn.Module):

    def __init__(self, agent: Backbone, env_model: EnvApp, device: str, agent_args : BasicArgs) -> None:
        super().__init__()

        self.agent = agent.to(device=device)
        self.env_model = env_model.to(device=device)
        self.device = device

        self.colors = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        self.agent_args = agent_args

    def greedy_agent_action(self, obs: torch.Tensor) -> np.ndarray:

        # get last frame
        obs = obs[:, -1]

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).permute(
            0, 3, 1, 2
        )

        with torch.no_grad():

            return torch.argmax(self.agent(obs), -1).cpu().numpy() - 1

    def pred_env(self, obs: torch.Tensor, actions: torch.Tensor, with_raw = False) -> torch.Tensor:

        # obs of length 3

        if obs.shape[1] != 3 and len(obs.shape) != 5:
            raise ValueError("Expected as batch")

        obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)

        # blacken border
        obs[:, :, [0, -1]] = 0
        obs[:, :, :, [0, -1]] = 0

        action_mask = torch.zeros(
            (obs.shape[0], 16, 16, 3), dtype=torch.float32, device=self.device
        )
        action_mask[:, :, :, actions + 1] = 1

        x = torch.cat([obs[:, 0], obs[:, 1], obs[:, 2], action_mask], dim=-1)
        x = x.permute(0, 3, 2, 1)

        with torch.no_grad():
            pred = torch.argmax(self.env_model(x), dim=1)
        
        img = self.colors[pred].transpose(1, 2)
        img[:, [0, -1]] = 0.5
        img[:, :, [0, -1]] = 0.5

        bs = obs.shape[0]
        flattened = pred.view(bs, -1)

        rewards = torch.zeros(bs, dtype=torch.float32, device=self.device)
        
        # if no red than -> loss
        mask = (flattened == 2).any(1)
        
        rewards = torch.where(mask, rewards, -1)
        
        # if not loss and no white -> win
        mask = torch.logical_and(mask, ~(flattened == 3).any(1))
        rewards = torch.where(mask, 1, rewards)
        
        if with_raw:
            return img.cpu(), rewards.cpu(), pred
            
            
        return img.cpu(), rewards.cpu()
    
    def env_model_performance(self, n_steps):
        
        unique_vals = self.colors.cpu().numpy()
        
        
        def com_label_acc(img, obs, reward):
            
            label = obs[-1]
            label[:, [0, -1]] = 0
            label[[0, -1], :] = 0

            label = (
                (label[:, :, None, :] == unique_vals[None, None, :])
                .all(-1)
                .astype(np.int32)
            )

            if reward == 1:
                label[label[:, :, 1] == 1, 0] = 1
                label[label[:, :, 1] == 1, 1] = 0

            elif reward == -1:
                label[:, :, 0] = 1
                label[:, :, 1:] = 0
                
            label = np.argmax(label, -1).T
                
            return (img[0].numpy() == label).all()

        
        env = SnakeGame(width=self.agent_args.width_and_height,
                height=self.agent_args.width_and_height,
                border=self.agent_args.border,
                food_amount=1,
                render_mode="human",
                manhatten_fac = 0,
                mode = "eval",
                seed=0 + self.agent_args.seed,)

        env = FrameStack(env)
    
        ar_acc = 0
        label_acc = 0
        ar_counter = 0
        reward_acc = 0
        
        i = 0
        
        def start():
            obs = env.reset()[0]
            
            for _ in range(3):
                action = self.greedy_agent_action(obs[None, :])

                obs, reward, done, truncated, info = env.step(action[0])
        
            return obs
        
        obs = start()
        
        for _ in tqdm(range(n_steps), desc = "Evaluating Env Model performance"):
            
            action = self.greedy_agent_action(obs[None, :])
            ar, re, pred = self.pred_env(obs[None, :], action, with_raw = True)
            
            obs, reward, done, truncated, info = env.step(action[0])
            
            reward_acc += re[0] == reward
            if reward == 0:
                ar_acc += np.all(ar.numpy()[0] == obs[-1])
                ar_counter += 1
                label_acc += com_label_acc(pred, obs, reward)
            
            if done:
                obs = start()
            
            i += 1
            
        return ar_acc/ar_counter, label_acc/ ar_counter, reward_acc / n_steps


class Wrapper(LightningModule):

    def __init__(self, args: NewArgs, agent: DoubleQNET, agent_args):
        super().__init__()
        self.args = args
        self.model = EnvApp(args)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, args.loss_weight, 1, args.loss_weight]))
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=4)
        self.mean_loss_metric = MeanMetric()
        self.success_rate_metric = MeanMetric()
        self.sampler = MarkovSampler(agent.model, self.model, args.device,  agent_args)
        

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.loss_fn(logits, y)

        mean_loss = self.mean_loss_metric(loss)
        self.log("train_loss", mean_loss, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy_metric(preds, y)
        self.log("train_accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)

        success_rate = self.success_rate_metric(self.succes_rate(preds, y))
        self.log("train_success_rate", success_rate, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def succes_rate(self, preds, y):
        
        val = (preds == y).view(preds.shape[0], -1).all(dim = 1).to(torch.float32)
        return val

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        mean_loss = self.mean_loss_metric(loss)
        self.log("val_loss", mean_loss, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy_metric(preds, y)
        self.log("val_accuracy", acc, prog_bar=True)

        success_rate = self.success_rate_metric(self.succes_rate(preds, y))
        self.log("val_success_rate", success_rate, prog_bar=True)

    
    def on_validation_epoch_end(self):
        succ, imgacc, rew = self.sampler.env_model_performance(self.args.val_steps)
        self.log("sampling/succ", succ, prog_bar = True)
        self.log("sampling/imgacc", imgacc, prog_bar = False)
        self.log("sampling/reward", rew, prog_bar = True)
        
        
        

def train(args: NewArgs):

    agent_args = BasicArgs(
            seed=0,
            n_envs=8,
            manhatten_fac=0,
            batch_size=32,
            min_eps=0.05,
            n_epoch_refil=1,
            n_obs_reward=args.n_steps_fut + 1,
            device = args.device
        )

    random.seed(agent_args.seed)
    np.random.seed(agent_args.seed)
    torch.manual_seed(agent_args.seed)

    agent = DoubleQNET(agent_args)
    agent.load_state_dict(torch.load("best.ckpt", map_location=agent_args.device))

    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                width=agent_args.width_and_height,
                height=agent_args.width_and_height,
                border=agent_args.border,
                food_amount=agent_args.food_amount,
                render_mode="rgb_array",
                manhatten_fac=agent_args.manhatten_fac,
                mode="train",
                seed=i + agent_args.seed,
            )
            for i in range(agent_args.n_envs)
        ]
    )

    data = (
        gen_data(args.data_points, 0.2, agent=agent, agent_args=agent_args,envs=envs, args=args)
        + gen_data(args.data_points, 0.1, agent=agent, agent_args=agent_args,envs=envs, args=args)
        + gen_data(args.data_points, 0.05, agent=agent, agent_args=agent_args,envs=envs, args=args)
        + gen_data(args.data_points, 0, agent=agent, agent_args=agent_args,envs=envs, args=args)
    )

    np.random.shuffle(data)
    val_data = data[: int(len(data) * 0.2)]
    train_data = data[int(len(data) * 0.2) :]

    train_dataset = EnvDataset(train_data)
    val_dataset = EnvDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.n_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batchsize, num_workers=args.n_workers
    )

    wrapper = Wrapper(args, agent, agent_args)

    if args.test:
        args.run_name += "_test"
        args.wandb_mode = "offline"

    ckpt_path = os.path.join("ckpt", args.run_name)
    os.makedirs(ckpt_path, exist_ok=True)

    loggers = [
        WandbLogger(
            project="env_approximation",
            name=args.run_name,
            log_model=True,
            config = args.model_dump(),
            mode = args.wandb_mode   
        ),
        
    ]

    trainer = Trainer(
        accelerator=args.device,
        fast_dev_run=args.test,
        max_epochs=args.epochs,
        logger=loggers,
        callbacks=ModelCheckpoint(
            ckpt_path, monitor="val_success_rate", mode="max", save_last=True
        ),
    )

    trainer.fit(wrapper, train_loader, val_loader)

def main():

    with open(sys.argv[1], "r") as f:

        args = NewArgs(**yaml.safe_load(f))

    train(args)
    
if __name__ == "__main__":
    main()