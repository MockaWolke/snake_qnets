from snake_game import *
import numpy as np
import gymnasium as gym
import torch
import random
from tqdm import tqdm
import os
from typing import Optional, List, Tuple, Set, Dict
from train_args import BasicArgs, parse_args
from models import DoubleQNET
from buffer import sample_eps_greedy
from torch.utils.data import Dataset, DataLoader
from lightning import LightningModule, Trainer
from pydantic import BaseModel
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, MeanMetric
from snake_game import make_stack_env, FrameStack
import gdown
import yaml
import sys
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt

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
    loss_weight: float = 10.0
    val_steps: int = 1000


unique_vals = np.array(
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=np.float32,
)

def preprocess_input(obs, actions, device):
    
    bs = obs.shape[0]
    obs = torch.tensor(obs, device=device, dtype=torch.float32)
    actions = torch.tensor(actions, device=device, dtype=torch.long)

    actions *= -1

    action_mask = torch.zeros(
        (obs.shape[0], 16, 16, 3), dtype=torch.float32, device=device
    )
    action_mask[torch.arange(bs), :, :, actions + 1] = 1

    assert (action_mask.sum(-1) == 1).all()

    x = torch.cat([obs[:, 0], obs[:, 1], obs[:, 2], action_mask], dim=-1)
    
    return x


def preprocess_label(label, reward):
    
    label = label.copy()
    label[:, [0, -1]] = 0
    label[[0, -1], :] = 0
    
    label = (
            (label[:, :, None, :] == unique_vals[None, None, :])
            .all(-1)
            .astype(np.int32)
        )
    
    label = np.argmax(label, -1)

    if reward == 1:
        mask = label == 1 
        label[mask] = 0

    # everything false if loss
    elif reward == -1:
        label[:] = 0
        
    return label

def pred_to_img(pred, colors, device):
    
    bs = pred.shape[0]
    
    img = colors[pred]
    img[:, [0, -1]] = 0.5
    img[:, :, [0, -1]] = 0.5
        

    flattened = pred.reshape(bs, -1)

    rewards = torch.zeros(bs, dtype=torch.float32, device=device)

    # if no red than -> loss
    mask = (flattened == 2).any(1)

    rewards = torch.where(mask, rewards, -1)

    # if not loss and no green -> win
    mask = torch.logical_and(mask, ~(flattened == 1).any(1))
    rewards = torch.where(mask, 1, rewards)
    
    return img, rewards

def preprocess(obs, label, action, reward):

    diffs = (obs[:-1] != obs[1:]).any(-1).sum((1, 2))
    if not np.isin(diffs, [0, 2, 3]).all():
        raise ValueError(f"logic fucked {diffs}")
        
    x = preprocess_input(obs[None,:], torch.tensor([action])[None,:], "cpu").numpy()[0]

    y = preprocess_label(label, reward)

    # if reward==-1:
    
    #     fig = plt.figure(figsize=(10,4))
        
    #     for i in range(3):
    #         plt.subplot(1, 4, i + 1)
    #         plt.imshow(obs[i])
    #         plt.axis("off")
        
    #     plt.subplot(1, 4, 4)
    #     plt.imshow(label)
    #     plt.axis("off")
    #     plt.title(f"{action} - {reward}")
    #     plt.show()

        

    return x, y


def gen_data(size, eps, agent, envs, agent_args, args):
    
    colors = torch.tensor(unique_vals)
    
    data = []

    samples = []

    steps = int(np.ceil(size / agent_args.n_envs))


    def sample_func(env, obs):
        return agent.get_greedy_action(obs)

    old_obs = envs.reset()[0]

    for _ in tqdm(
        range(steps + agent_args.n_obs_reward - 1), desc=f"sample for eps {eps}"
    ):

        actions = sample_func(envs, old_obs[:, -1])

        actions = sample_eps_greedy(actions, eps)

        new_obs, rewards, terminated, _, scores = envs.step(actions)
        labels = new_obs[:, -1]

        for obs, label, action, reward in zip(old_obs, labels, actions, rewards):



            x, y= preprocess(obs, label, action, reward)
            
            img, re = pred_to_img(torch.tensor(y)[None, :], colors, "cpu")
            img = img[0].numpy()
            
            if reward == 0 and (not (img == label).all() or re[0]!= reward):
                raise ValueError("Hey", not (img == label).all())
                
                
            elif reward == 1 and (not (img != label).any(-1).sum() == 1 or re[0]!= reward):
                raise ValueError("win false", (img != label).sum())
            
            elif reward == -1 and re[0]!= reward:
                raise ValueError(reward, re)
                
            samples.append((x, y))

        old_obs = new_obs
    return samples


class EnvDataset(Dataset):

    def __init__(self, data) -> None:
        super().__init__()

        self.data = data

    def __getitem__(self, index):

        x, y = self.data[index]
        
        x = torch.tensor(x, dtype=torch.float32)

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
        self.block2 = ConvBlock(w, w * 2)

        self.pool = nn.MaxPool2d(2)
        self.smaller_block1 = ConvBlock(w * 2, w * 4)
        self.smaller_block2 = ConvBlock(w * 4, w * 2)

        self.inv = nn.ConvTranspose2d(
            w * 2, w * 2, 3, padding=1, output_padding=1, stride=2
        )

        self.block3 = ConvBlock(w * 4, w * 2)
        self.block4 = ConvBlock(w * 2, w * 2)

        self.out_layer = nn.Conv2d(w * 2, 4, 1, padding=0)

    def forward(self, x):
        
        x = x.permute(0, 3, 1, 2)

        x = self.block1(x)
        x = self.block2(x)

        smaller = self.pool(x)
        smaller = self.smaller_block1(smaller)
        smaller = self.smaller_block2(smaller)

        x = torch.cat([x, self.inv(smaller)], dim=1)

        x = self.block3(x)
        x = self.block4(x)

        return self.out_layer(x)


class MarkovSampler(nn.Module):

    def __init__(
        self, agent, env_model: EnvApp, device: str, agent_args: BasicArgs
    ) -> None:
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

    def env_model_performance(self, n_steps):


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

        for _ in tqdm(range(n_steps), desc="Evaluating Env Model performance"):

            action = self.greedy_agent_action(obs[None, :].copy())
            
            # plt.figure(figsize=(10,4))
            # for i,img in enumerate(obs):
                # plt.subplot(1,5,1 + i)
                # plt.imshow(img)
                # plt.axis("off")
                
                
            ar, re, pred = self.pred_env(obs[None, :].copy(), action*-1, with_raw=True)

            # plt.subplot(1,5,4)
            # plt.imshow(ar[0])
            # plt.axis("off")

            obs, reward, done, truncated, info = env.step(action[0])
            
            # plt.subplot(1,5,5)
            # plt.imshow(obs[-1])
            # plt.axis("off")
            # plt.title(str(action))
            # plt.show()

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


class Wrapper(LightningModule):

    def __init__(self, args: NewArgs, agent: DoubleQNET, agent_args):
        super().__init__()
        self.args = args
        self.model = EnvApp(args)
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1, args.loss_weight, 1, args.loss_weight])
        )
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=4)
        self.mean_loss_metric = MeanMetric()
        self.success_rate_metric = MeanMetric()
        self.sampler = MarkovSampler(agent.model, self.model, args.device, agent_args)

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
        self.log(
            "train_success_rate",
            success_rate,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def succes_rate(self, preds, y):

        val = (preds == y).view(preds.shape[0], -1).all(dim=1).to(torch.float32)
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
        succ, imgacc, rew, score = self.sampler.env_model_performance(self.args.val_steps)
        self.log("sampling/succ", succ, prog_bar=True)
        self.log("sampling/imgacc", imgacc, prog_bar=False)
        self.log("sampling/reward", rew, prog_bar=True)
        self.log("sampling/score", score, prog_bar=True)


def train(args: NewArgs):

    agent_args = BasicArgs(
        n_envs=8,
        manhatten_fac=0,
        batch_size=32,
        min_eps=0.05,
        n_epoch_refil=1,
        n_obs_reward=args.n_steps_fut + 1,
        device=args.device,
    )

    random.seed(agent_args.seed)
    np.random.seed(agent_args.seed)
    torch.manual_seed(agent_args.seed)

    agent = DoubleQNET(agent_args)
    agent.load_state_dict(torch.load("best.ckpt", map_location=agent_args.device))

    envs = gym.vector.AsyncVectorEnv(
        [
            make_stack_env(
                num_stacks=3,
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
        gen_data(
            args.data_points,
            0.2,
            agent=agent,
            agent_args=agent_args,
            envs=envs,
            args=args,
        )
        + gen_data(
            args.data_points,
            0.1,
            agent=agent,
            agent_args=agent_args,
            envs=envs,
            args=args,
        )
        + gen_data(
            args.data_points,
            0.05,
            agent=agent,
            agent_args=agent_args,
            envs=envs,
            args=args,
        )
        + gen_data(
            args.data_points,
            0,
            agent=agent,
            agent_args=agent_args,
            envs=envs,
            args=args,
        )
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
            config=args.model_dump(),
            mode=args.wandb_mode,
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
