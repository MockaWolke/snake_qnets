import numpy as np
from numpy.random import randint
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class SnakeGameEnv(gym.Env):
    """Snake game environment compatible with Gymnasium (OpenAI Gym interface)."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15
    }

    def __init__(self, width=20, height=20, food_amount=1, border=0, grass_growth=0, max_grass=0, render_mode=None):
        super(SnakeGameEnv, self).__init__()
        self.width = width
        self.height = height
        self.food_amount = food_amount
        self.border = border
        self.grass_growth = grass_growth
        self.max_grass = max_grass
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.height, self.width, 3), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # Actions: -1 (left), 0 (straight), 1 (right)

        self.board = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.grass = np.zeros((self.height, self.width)) + max_grass
        self.window = None
        self.clock = None
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        self.score = 0
        self.done = False
        self.direction = randint(0, 4)
        self.snake = []
        self.apples = []
        self.grass[:, :] = self.max_grass
        self.create_snake()
        self.create_apples()
        return self.board_state(), {}

    def step(self, action):
        """Move snake/game one step based on action."""
        direction_change = int(action) - 1
        self.direction = (self.direction + direction_change) % 4
        self.grow_snake(self.direction)

        reward = 0
        if self.snake[0] in self.apples:
            self.apples.remove(self.snake[0])
            reward = 1
            self.create_apples()
        else:
            self.snake.pop()
            self.check_collisions()
            if self.done:
                terminated = True
                reward = -1
            else:
                terminated = False

        if reward >= 0:
            x, y = self.snake[0]
            reward += self.grass[x, y]
            self.grass[x, y] = 0
            self.score += reward
            self.grass += self.grass_growth
            self.grass[self.grass > self.max_grass] = self.max_grass

        observation = self.board_state()
        return observation, reward, terminated, False, {"score": self.score}

    def create_apples(self):
        """Create new apples away from the snake."""
        while len(self.apples) < self.food_amount:
            apple = (randint(0, self.height), randint(0, self.width))
            while apple in self.snake:
                apple = (randint(0, self.height), randint(0, self.width))
            self.apples.append(apple)

    def create_snake(self):
        """Create a snake of size 3 at a random position and orientation."""
        x = randint(5, self.width - 5)
        y = randint(5, self.height - 5)
        self.snake = []
        for _ in range(5):
            if self.direction == 0:
                y += 1
            elif self.direction == 1:
                x -= 1
            elif self.direction == 2:
                y -= 1
            elif self.direction == 3:
                x += 1
            self.snake.append((y, x))

    def grow_snake(self, d):
        """Add one position to snake head."""
        y, x = self.snake[0]
        if d == 0:
            y -= 1
        elif d == 1:
            x += 1
        elif d == 2:
            y += 1
        else:
            x -= 1
        self.snake.insert(0, (y, x))

    def check_collisions(self):
        """Check if game is over by colliding with edge or itself."""
        y, x = self.snake[0]
        if x == -1 or x == self.width or y == -1 or y == self.height or (y, x) in self.snake[1:]:
            self.done = True

    def board_state(self):
        """Render the environment state."""
        self.board[:, :, :] = 0
        if self.max_grass > 0:
            self.board[:, :, 1] = self.grass / self.max_grass * 0.3
        if not self.done:
            y, x = self.snake[0]
            self.board[y, x, :] = 1
        for y, x in self.snake[1:]:
            self.board[y, x, 0] = 1
        for y, x in self.apples:
            self.board[y, x, 1] = 1
        return self.board

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self.board_state()
        elif self.render_mode == "human":
            plt.imshow(self.board_state(), interpolation="nearest")
            plt.axis("off")
            plt.show(block=False)
            plt.pause(1 / self.metadata["render_fps"])

    def close(self):
        """Close the environment."""
        if self.window is not None:
            plt.close(self.window)
            self.window = None

def make_env(seed, **kwargs):
    def _init():
        env = SnakeGameEnv(**kwargs)
        env.reset(seed=seed)
        return env

    return _init


if __name__ == '__main__':
    game = SnakeGameEnv(20,20, render_mode = "human")
    game.render()
    
			
