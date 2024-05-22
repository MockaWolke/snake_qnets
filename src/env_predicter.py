from snake_game import *
import numpy as np
import gymnasium as gym
import torch
import random
from tqdm import tqdm
import os
from train_args import BasicArgs
from models import DoubleQNET
from torch.utils.data import DataLoader
from lightning import LightningModule, Trainer
import torch.nn as nn
from torchmetrics import Accuracy, MeanMetric
import yaml
import sys
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from env_utils import EnvApp, EnvDataset, NewArgs, gen_data
from markov_sampler import MarkovSampler



class Wrapper(LightningModule):

    def __init__(self, args: NewArgs, agent: DoubleQNET, agent_args):
        super().__init__()
        self.args = args
        self.agent_args = agent_args
        self.model = EnvApp(args)
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1, args.loss_weight, 1, args.loss_weight])
        )
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=4)
        self.mean_loss_metric = MeanMetric()
        self.success_rate_metric = MeanMetric()
        self.agent_model = agent.model

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
        
        sampler = MarkovSampler(self.agent_model, self.model, self.args.device, self.agent_args)
        
        succ, imgacc, rew, score = sampler.eval_env_model_performance(self.args.val_steps)
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
    val_set = {i for i in val_dataset}
    
    for i in tqdm(train_dataset, desc="Checking overlap"):
        assert i not in val_set, "mix of data"


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

    torch.save(wrapper.model.state_dict(), os.path.join(ckpt_path, "manual.ckpt"))


def main():

    with open(sys.argv[1], "r") as f:

        args = NewArgs(**yaml.safe_load(f))

    train(args)


if __name__ == "__main__":
    main()
