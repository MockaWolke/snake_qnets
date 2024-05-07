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


class ReplayBuffer:

    def __init__(self, args: BasicArgs) -> None:
        self.replays = deque()
        self.args = args
        self.last_ops = None

    def fill(self, values):
        self.replays.extend(values)

    def sample(self, batchsize):

        sample = random.choices(self.replays, k=batchsize)

        new_obs, obs, reward, terminated, actions = map(np.array, zip(*sample))

        return new_obs, obs, reward, terminated, actions

    def init_fill(self, envs, heuristic):

        old_obs = envs.reset()[0]

        for _ in tqdm(
            range(int(self.args.buffer_size // self.args.n_envs)),
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

    def fill_epoch(self, envs, model: DoubleQNET):

        if len(self.replays) / self.args.buffer_size < 0.95:
            raise ValueError("Buffer not full")

        n_items = int(self.args.buffer_size / self.args.n_epoch_refill)

        for _ in range(n_items):
            self.replays.popleft()

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


def sample_eps_greedy(greedy_action, epsilon):

    mask = np.random.uniform(0, 1, greedy_action.shape) < epsilon

    random_actions = np.random.randint(-1, 2, greedy_action.shape)

    new_actions = np.where(mask, greedy_action, random_actions)

    if new_actions.min() < -1 or new_actions.max() > 1:
        print("the fuck", new_actions)
        raise ValueError()
    return new_actions


def evaluate_model(
    qnet: DoubleQNET, args: BasicArgs, logger: SummaryWriter, epoch: int, max_steps=1000
):

    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                width=args.width_and_height,
                height=args.width_and_height,
                border=args.border,
                food_amount=args.food_amount,
                render_mode="rgb_array",
                seed=i + args.seed,
            )
            for i in range(args.n_envs)
        ]
    )

    terminated = np.zeros(args.n_envs)
    rewards = np.zeros(args.n_envs)
    steps = np.zeros(args.n_envs)

    obs, _ = envs.reset()

    for _ in tqdm(range(max_steps), "Eval Model"):

        actions = qnet.get_greedy_action(obs)

        obs, reward, ter, _, scores = envs.step(actions)

        # add to reawrds when not terminated
        rewards += reward * (1-terminated)
        steps += (1-terminated)

        terminated = np.logical_or(terminated, ter)

        if terminated.all():
            break

    min_reward, max_reward, mean_reward, std_mean = map(
        float, (rewards.min(), rewards.max(), rewards.mean(), rewards.std())
    )
    
    min_step, max_step, mean_step, std_steps = map(
        float, (steps.min(), steps.max(), steps.mean(), steps.std())
    )
    
    

    logger.add_scalar("Eval/min_reward", min_reward, epoch)
    logger.add_scalar("Eval/max_reward", max_reward, epoch)
    logger.add_scalar("Eval/mean_reward", mean_reward, epoch)
    logger.add_scalar("Eval/min_step", min_step, epoch)
    logger.add_scalar("Eval/max_step", max_step, epoch)
    logger.add_scalar("Eval/mean_step", mean_step, epoch)

    print(f"Eval Results Epoch {epoch}:")
    print(f"\Rewards: min: {min_reward:.1f}, max: {max_reward:.1f}, mean: {mean_reward:.1f} +- {std_mean:.2f}")
    print(f"\tSteps: min: {min_step:.1f}, max: {max_step:.1f}, mean: {mean_step:.1f} +- {std_steps:.2f}")


def main(args: BasicArgs):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_path = os.path.join("logs", args.run_name)
    os.makedirs(log_path, exist_ok=True)

    logger = SummaryWriter(log_path)
    
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in args.model_dump().items()])),
    )

    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                width=args.width_and_height,
                height=args.width_and_height,
                border=args.border,
                food_amount=args.food_amount,
                render_mode="rgb_array",
                seed=i + args.seed,
            )
            for i in range(args.n_envs)
        ]
    )

    buffer = ReplayBuffer(args)
    buffer.init_fill(envs, closest_heuristic)

    qnet = DoubleQNET(args)
    
    global_step = args.init_step
    total_steps = args.epochs * args.steps_per_epoch
    initial_eps = args.eps
    initial_lr = args.lr
    
    print(f"Training for {total_steps} steps")

    for epoch in range(args.epochs):

        losses = []

        for _ in tqdm(
            range(args.steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}"
        ):

            batch = buffer.sample(args.batch_size)

            loss = qnet.step(batch)

            if global_step < args.eps_anneal_steps:
                args.eps = args.min_eps + (initial_eps - args.min_eps) * (1 - global_step / args.eps_anneal_steps)

            if args.min_lr is not None:
                args.lr = max(args.min_lr, initial_lr - (initial_lr - args.min_lr) * (global_step / total_steps))
                qnet.learning_rate = args.lr

            logger.add_scalar("StepLoss", loss, global_step)
            logger.add_scalar("Params/Epsilon", args.eps, global_step)
            logger.add_scalar("Params/Learning_Rate", qnet.learning_rate, global_step)            

            losses.append(loss.numpy())
            
            
            global_step += 1
            

        loss = np.mean(losses)
        logger.add_scalar("EpochLoss", loss, epoch)
        
        qnet.update_target_model()
        buffer.fill_epoch(envs, qnet)
        
        if epoch % args.eval_freq == 0:
            
            evaluate_model(qnet, args, logger, epoch, max_steps=args.max_eval_steps)
            
        if epoch % args.save_freq == 0:
            
            save_path = os.path.join(log_path, f"{epoch}.ckpt")
            print("saving to", save_path)
            
            torch.save(qnet.state_dict(), save_path)
        

if __name__ == "__main__":

    args = parse_args(BasicArgs)

    main(args)
