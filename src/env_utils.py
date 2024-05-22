from snake_game import *
import numpy as np
import torch
from tqdm import tqdm
from buffer import sample_eps_greedy
from torch.utils.data import Dataset
from pydantic import BaseModel
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F


def cast_to_tensor(t, device, dtype):
    
    if not torch.is_tensor(t):
    
        return torch.tensor(t, device=device, dtype=dtype)
    
    return t.to( device=device, dtype=dtype)
    

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
    
    obs = cast_to_tensor(obs, device, torch.float32)
    
    actions = cast_to_tensor(actions, device=device, dtype=torch.long)

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
