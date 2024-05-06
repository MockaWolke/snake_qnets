import gymnasium as gym
from snake_game import make_env

env = gym.vector.SyncVectorEnv([make_env(0) for _ in range(10)])

env.reset()

a = env.step(env.action_space.sample())[-1]

print(a)