import torch
import torch.nn as nn
import torch.nn.functional as F
from train_args import BasicArgs, UpdateModes
import copy
import numpy as np


def prepare_batch(batch, device):

    new_obs, obs, reward, terminated, actions, indices, weights = batch

    new_obs = torch.tensor(new_obs, dtype=torch.float32, device=device).permute(
        0, 3, 1, 2
    )
    obs = torch.tensor(obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    reward = torch.tensor(reward, dtype=torch.float32, device=device)
    terminated = torch.tensor(terminated, dtype=torch.bool, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device) + 1
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    return new_obs, obs, reward, terminated, actions, indices, weights


class SmallCNNBackbone(nn.Module):
    def __init__(self, input_channels, n_actions, imgsz=16):
        super(SmallCNNBackbone, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * imgsz**2, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DuelingCNNBackbone(nn.Module):
    def __init__(self, input_channels, n_actions, imgsz=16):
        super(DuelingCNNBackbone, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        self.value_head = nn.Sequential(
            nn.Linear(64 * imgsz**2, 512),
            nn.Linear(512, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(64 * imgsz**2, 512),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        value = self.value_head(x)
        adv = self.adv_head(x)
        adv = adv - adv.mean(axis=1, keepdim=True)

        return value + adv


class DoubleQNET(nn.Module):

    def __init__(self, args: BasicArgs):

        super(DoubleQNET, self).__init__()

        self.args = args

        self.imgsz = args.width_and_height + args.border * 2
        
        if args.backbone == "normal":
            
            self.model = SmallCNNBackbone(3, 3, self.imgsz).to(args.device)
        elif args.backbone == "dueling":
            self.model = DuelingCNNBackbone(3, 3, self.imgsz).to(args.device)
            
        else:
            raise ValueError("wrong backbone")

        self.optim = torch.optim.Adam(self.model.parameters(), args.lr)

        self.target_model = copy.deepcopy(self.model)
        self.criterion = torch.nn.HuberLoss(reduction="none")

    @property
    def learning_rate(self):
        return self.optim.param_groups[0]["lr"]

    @learning_rate.setter
    def learning_rate(self, value):
        self.optim.param_groups[0]["lr"] = value

    def update_target_model(self, step):

        # if hard copy
        if (
            self.args.update_mode == UpdateModes.epoch_copy
            and step % self.args.steps_per_epoch == 0
        ):
            self.target_model = copy.deepcopy(self.model)

        elif (
            self.args.update_mode == UpdateModes.polyak
            and step % self.args.update_freq == 0
        ):
            self.polyak_average()

    def polyak_average(self):
        with torch.no_grad():
            for target_param, local_param in zip(
                self.target_model.parameters(), self.model.parameters()
            ):
                target_param.data.copy_(
                    (1 - self.args.theta) * target_param.data
                    + self.args.theta * local_param.data
                )

    def step(self, batch, update_func):

        new_obs, obs, reward, terminated, actions, indices, weights = prepare_batch(
            batch, self.args.device
        )

        pred_vals = torch.gather(self.model(obs), 1, actions[:, None]).squeeze()

        with torch.no_grad():
            next_vals = torch.max(self.target_model(new_obs), -1).values

        # if terminated ignore next values
        next_vals *= 1 - terminated.to(next_vals.dtype)

        y_true = reward + self.args.gamma * next_vals

        loss = (self.criterion(y_true, pred_vals) * weights).mean()

        self.optim.zero_grad()

        loss.backward()

        self.optim.step()

        tde_error = (y_true - pred_vals).detach().cpu().numpy()

        update_func(indices, tde_error)

        return loss.detach().cpu()

    def get_greedy_action(self, obs):

        obs = torch.tensor(obs, dtype=torch.float32, device=self.args.device).permute(
            0, 3, 1, 2
        )

        with torch.no_grad():

            preds = torch.argmax(self.model(obs), -1).cpu().numpy()

        # for proper range
        return preds - 1
