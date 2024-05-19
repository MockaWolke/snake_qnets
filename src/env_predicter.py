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

        _, _, _, _, actions = samples[idx]

        s_reward, s_terminated, old_obs = [], [], []

        for i in range(agent_args.n_obs_reward):

            new_obs, old, reward, terminated, _ = samples[
                idx + i
            ]  # get respective future points

            s_reward.append(reward)
            s_terminated.append(terminated)
            old_obs.append(old)

        s_reward, s_terminated, old_obs = map(
            lambda x: np.stack(x, axis=1), (s_reward, s_terminated, old_obs)
        )

        for tup in zip(old_obs, s_reward, s_terminated, actions):
            old_obs, s_reward, s_terminated, actions = tup
            old_obs[:, [0, -1]] = 0
            old_obs[:, :, [0, -1]] = 0
            old_obs, label = old_obs[: args.n_steps_fut], old_obs[-1]
            reward = s_reward[-2]

            if s_terminated[:-2].any():
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


class EnvApp(Module):

    def __init__(self, args: NewArgs) -> None:

        super().__init__()

        w = args.m_width

        self.conv1 = nn.Conv2d((args.n_steps_fut + 1) * 3, w, 3, padding=1)

        self.conv2 = nn.Conv2d(w, w, 3, padding=1)

        self.conv3 = nn.Conv2d(w, w, 3, padding=1)
        self.conv4 = nn.Conv2d(w, w, 3, padding=1)

        self.norm1 = nn.BatchNorm2d(w)
        self.norm2 = nn.BatchNorm2d(w)
        self.norm3 = nn.BatchNorm2d(w)
        self.norm4 = nn.BatchNorm2d(w)

        self.pool = nn.MaxPool2d(2)
        self.smaller_conv1 = nn.Conv2d(w, w * 2, 3, padding=1)
        self.smaller_conv2 = nn.Conv2d(w * 2, w * 2, 3, padding=1)
        self.s_norm1 = nn.BatchNorm2d(w * 2)
        self.s_norm2 = nn.BatchNorm2d(w * 2)

        self.inv = nn.ConvTranspose2d(
            w * 2, w, 3, padding=1, output_padding=1, stride=2
        )

        self.out_layer = nn.Conv2d(w, 4, 1, padding=0)

    def forward(self, x):

        x = F.relu(self.norm1(self.conv1(x)))
        x = x + F.relu(self.norm2(self.conv2(x)))
        bigger = x + F.relu(self.norm3(self.conv3(x)))

        smaller = F.relu(self.s_norm1(self.smaller_conv1(self.pool(bigger))))
        smaller = smaller + F.relu(self.s_norm2(self.smaller_conv2(smaller)))

        x = bigger + self.inv(smaller)

        x = x + F.relu(self.norm4(self.conv4(x)))

        return self.out_layer(x)


class Wrapper(LightningModule):

    def __init__(self, args: NewArgs):
        super().__init__()
        self.args = args
        self.model = EnvApp(args)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=4)
        self.mean_loss_metric = MeanMetric()
        self.success_rate_metric = MeanMetric()

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
        self.log("train_accuracy", acc, prog_bar=True)

        success_rate = self.success_rate_metric(self.succes_rate(preds, y))
        self.log("train_success_rate", success_rate, prog_bar=True)

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


def main():

    with open(sys.argv[1], "r") as f:

        args = NewArgs(**yaml.safe_load(f))

    agent_args = BasicArgs(
        seed=0,
        n_envs=8,
        manhatten_fac=0,
        batch_size=32,
        min_eps=0.05,
        n_epoch_refil=1,
        n_obs_reward=args.n_steps_fut + 1,
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

    wrapper = Wrapper(args)

    if args.test:
        args.run_name += "_test"

    ckpt_path = os.path.join("ckpt", args.run_name)
    os.makedirs(ckpt_path, exist_ok=True)

    loggers = [
        WandbLogger(
            project="env_approximation",
            name=args.run_name,
            offline=args.test,
            log_model=True,
            config = args.model_dump(),
             
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

if __name__ == "__main__":
    main()