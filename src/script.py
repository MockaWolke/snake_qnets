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





def evaluate_model(
    qnet: DoubleQNET, args: BasicArgs, writer: SummaryWriter, epoch: int, max_steps=1000
):

    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                width=args.width_and_height,
                height=args.width_and_height,
                border=args.border,
                food_amount=1,
                render_mode="rgb_array",
                manhatten_fac = 0,
                mode = "eval",
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
        
        if (reward - np.floor(reward)).any():
            print("None int rewards probably error in eval")
        
        # add to reawrds when not terminated
        rewards += reward * (1 - terminated)
        steps += 1 - terminated

        terminated = np.logical_or(terminated, ter)

        if terminated.all():
            break

    min_reward, max_reward, mean_reward, std_mean = map(
        float, (rewards.min(), rewards.max(), rewards.mean(), rewards.std())
    )

    min_step, max_step, mean_step, std_steps = map(
        float, (steps.min(), steps.max(), steps.mean(), steps.std())
    )

    writer.add_scalar("Eval/min_reward", min_reward, epoch)
    writer.add_scalar("Eval/max_reward", max_reward, epoch)
    writer.add_scalar("Eval/mean_reward", mean_reward, epoch)
    writer.add_scalar("Eval/min_step", min_step, epoch)
    writer.add_scalar("Eval/max_step", max_step, epoch)
    writer.add_scalar("Eval/mean_step", mean_step, epoch)

    print(f"Eval Results Epoch {epoch}:")
    print(
        f"\tRewards: min: {min_reward:.1f}, max: {max_reward:.1f}, mean: {mean_reward:.1f} +- {std_mean:.2f}"
    )
    print(
        f"\tSteps: min: {min_step:.1f}, max: {max_step:.1f}, mean: {mean_step:.1f} +- {std_steps:.2f}"
    )

    return mean_reward



def train(args : BasicArgs, log_path, ckpt_path):
    writer = SummaryWriter(log_path)

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in args.model_dump().items()])),
    )

    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                width=args.width_and_height,
                height=args.width_and_height,
                border=args.border,
                food_amount=args.food_amount,
                render_mode="rgb_array",
                manhatten_fac = args.manhatten_fac,
                mode = "train",
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
    best_score = -np.inf

    logger.success(f"Training for {total_steps} steps")


    for epoch in range(args.epochs):
        losses = []

        for _ in tqdm(
            range(args.steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}"
        ):
            batch = buffer.sample(args.batch_size)

            loss = qnet.step(batch, buffer.update)

            if global_step < args.eps_anneal_steps:
                args.eps = args.min_eps + (initial_eps - args.min_eps) * (
                    1 - global_step / args.eps_anneal_steps
                )

            if args.min_lr is not None:
                args.lr = max(
                    args.min_lr,
                    initial_lr
                    - (initial_lr - args.min_lr) * (global_step / total_steps),
                )
                qnet.learning_rate = args.lr
                
                
            args.buffer_beta = min(1.0, args.buffer_beta + args.incr_buffbeta)

            writer.add_scalar("StepLoss", loss, global_step)
            writer.add_scalar("Params/Epsilon", args.eps, global_step)
            writer.add_scalar(
                "Params/Learning_Rate", qnet.learning_rate, global_step
            )
            writer.add_scalar("Params/Beta", args.buffer_beta, global_step)

            losses.append(loss.numpy())

            global_step += 1
            # call this every step for polyiak avering
            qnet.update_target_model(global_step)

        loss = np.mean(losses)
        writer.add_scalar("EpochLoss", loss, epoch)

        buffer.fill_epoch(envs, qnet)

        if epoch % args.eval_freq == 0:
            new_score = evaluate_model(
                qnet, args, writer, epoch, max_steps=args.max_eval_steps
            )

            save_path = os.path.join(ckpt_path, "last.ckpt")
            print("saving to", save_path)
            torch.save(qnet.state_dict(), save_path)

            if new_score > best_score:
                print("best result -> saving as best.ckpt")
                shutil.copyfile(save_path, os.path.join(ckpt_path, "best.ckpt"))

            best_score = max(new_score, best_score)

        # final evaluation with more envs
    args.n_envs *= 3
    evaluate_model(qnet, args, writer, args.epochs, max_steps=args.max_eval_steps)
    return qnet



def main(args: BasicArgs):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_path = os.path.join("logs", args.run_name)
    ckpt_path = os.path.join("ckpt", args.run_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    if args.use_wandb:
        logger.info("Using Wandb!")
        wandb.tensorboard.patch(root_logdir=log_path)
        wandb.init(
            project="snake",
            config=args.model_dump(),
            group=args.run_group,
            save_code=True,
            name=args.run_name,
            mode=args.wandb_mode,
            sync_tensorboard=True,
        )

    try:
        qnet = train(args, log_path, ckpt_path)
    except Exception:
        logger.exception("Training Failed")
    
    try:
        
        if args.make_vid:
            vid_path = os.path.join(log_path, "run.gif")
            logger.info("creating video at ", vid_path)
            make_vid(vid_path, qnet, args, min_steps=30, max_steps=500, max_tries = 4)
            
            if args.use_wandb: wandb.log({"agent": wandb.Video(vid_path, fps=10, format="gif")})
                
    except Exception:
        logger.exception("Vid Save Failed")
        
    if args.use_wandb:
        wandb.save(os.path.join(ckpt_path, "*.ckpt"))

if __name__ == "__main__":

    args = parse_args(BasicArgs)

    main(args)
