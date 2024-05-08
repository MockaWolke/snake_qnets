from models import DoubleQNET
from train_args import BasicArgs
from snake_game import SnakeGame
import numpy as np
import torch
from PIL import Image, ImageDraw

def sample_run(model, args: BasicArgs, min_steps=10, max_steps=200, max_tries = 4):

    env = SnakeGame(
        width=args.width_and_height,
        height=args.width_and_height,
        border=args.border,
        food_amount=args.food_amount,
        render_mode="rgb_array",
        manhatten_fac=0,
        mode="eval",
    )

    n = 0
    while True:
        steps = 0
        obs, _ = env.reset()
        arrays = [obs]
        cum_reward = 0

        rewards = [cum_reward]
        terminated = False

        while not terminated:

            obs, reward, terminated, _, score = env.step(
                model.get_greedy_action(obs[None, :])[0]
            )

            arrays.append(obs)
            cum_reward += reward
            rewards.append(cum_reward)

            if terminated or steps > max_steps:
                break

            steps += 1

        if steps >= min_steps or n==max_tries:
            return arrays, rewards

        n+= 1

def make_vid(path, model, args: BasicArgs, min_steps=10, max_steps=200, fps=10, max_tries = 4):

    res = sample_run(model, args, min_steps=min_steps, max_steps=max_steps, max_tries = max_tries)

    frames = []

    for frame, reward in zip(*res):

        frame = np.repeat(np.repeat(frame, 4, axis=0), 4, axis=1)
        frame = Image.fromarray((frame * 255).astype(np.uint8))
        draw = ImageDraw.Draw(frame)
        draw.text((5, 5), str(int(reward)))

        frames.append(frame)

    duration_per_frame = int(1000 / fps)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration_per_frame,
        loop=0,
    )