import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

int_to_dir = {0:"up", 1:'right', 2:'down',3:'left'}
dir_to_int = dict(zip(int_to_dir.values(), int_to_dir.keys()))


def manhatten(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class SnakeGame(gym.Env):
    "Implements the snake game core"
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        width=14,
        height=14,
        food_amount=1,
        border=1,
        grass_growth=0,
        max_grass=0,
        render_mode=None,
        manhatten_fac = 0,
        seed = None,
        mode = "train",
    ):
        "Initialize board"
        super(SnakeGame, self).__init__()

        if mode != "train" and (manhatten_fac!=0 or food_amount!= 1):
            
            raise ValueError(f"not working with mode {mode}")
                

        self.width = width
        self.height = height
        self.board = np.zeros((height, width, 3), dtype=np.float32)
        self.food_amount = food_amount
        self.border = border
        self.grass_growth = grass_growth
        self.grass = np.zeros((height, width)) + max_grass
        self.max_grass = max_grass
        self.manhatten_fac = manhatten_fac

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.height + border * 2, self.width + border * 2, 3),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3, start=-1)

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.reset()
        
        self.seed(seed)
        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def create_apples(self):
        "create a new apple away from the snake"
        while len(self.apples) < self.food_amount:
            apple = (self.np_random.integers(0, self.height - 1), self.np_random.integers(0, self.width - 1))
            while apple in self.snake:
                apple = (self.np_random.integers(0, self.height - 1), self.np_random.integers(0, self.width - 1))
            self.apples.append(apple)

    def create_snake(self):
        "create a snake, size 3, at random position and orientation"
        x = self.np_random.integers(5, self.width - 5)  # not t0o close to border
        y = self.np_random.integers(5, self.height - 5)
        self.direction = self.np_random.integers(0, 4)
        self.snake = []
        for i in range(5):
            if self.direction == 0:
                y = y + 1
            elif self.direction == 1:
                x = x - 1
            elif self.direction == 2:
                y = y - 1
            elif self.direction == 3:
                x = x + 1
            self.snake.append((y, x))

    def grow_snake(self, d):
        "add one position to snake head (0=up, 1=right, 2=down, 3=left)"
        y, x = self.snake[0]
        if d == 0:
            y = y - 1
        elif d == 1:
            x = x + 1
        elif d == 2:
            y = y + 1
        else:
            x = x - 1
        self.snake.insert(0, (y, x))

    def check_collisions(self):
        "check if game is over by colliding with edge or itself"
        # just need to check snake's head
        x, y = self.snake[0]
        if (
            x == -1
            or x == self.height
            or y == -1
            or y == self.width
            or (x, y) in self.snake[1:]
        ):
            self.done = True

    def step(self, action):
        """
        move snake/game one step
        action can be -1 (turn left), 0 (continue), 1 (turn rignt)
        """
        
        old_distance = min(manhatten(self.snake[0], apple) for apple in self.apples)
        
        direction = int(action)
        assert -1 <= direction <= 1
        self.direction += direction
        if self.direction < 0:
            self.direction = 3
        elif self.direction > 3:
            self.direction = 0
        self.grow_snake(self.direction)  
        
        new_distance = min(manhatten(self.snake[0], apple) for apple in self.apples)
        
        
        if self.snake[0] in self.apples:
            self.apples.remove(self.snake[0])
            reward = 1
            self.create_apples()  # new apple
        else:
            self.snake.pop()
            self.check_collisions()
            if self.done:
                reward = -1
            else:
                reward = (old_distance - new_distance) * self.manhatten_fac * min(5 / len(self.snake), 1.0)
        if reward >= 0:
            x, y = self.snake[0]
            reward += self.grass[x, y]
            self.grass[x, y] = 0
            self.score += reward
            self.grass += self.grass_growth
            self.grass[self.grass > self.max_grass] = self.max_grass

        return self.board_state(), reward, self.done, False, {"score": self.score}

    @property
    def state(self):
        "easily get current state (score, apple, snake head and tail)"
        score = self.score
        apple = self.apples
        head = self.snake[0]
        tail = self.snake[1:]
        return score, apple, head, tail, self.direction
    
    

    def print_state(self):
        "print the current board state"
        for i in range(self.height):
            line = "." * self.width
            for x, y in self.apples:
                if y == i:
                    line = line[:x] + "A" + line[x + 1 :]
            for s in self.snake:
                x, y = s
                if y == i:
                    line = line[:x] + "X" + line[x + 1 :]
            print(line)

    def test_step(self, direction):
        "to test: move the snake and print the game state"
        self.step(direction)
        self.print_state()
        if self.done:
            print("Game over! Score=", self.score)

    def reset(self):
        "reset state"
        self.score = 0
        self.done = False
        self.create_snake()
        self.apples = []
        self.create_apples()
        self.grass[:, :] = self.max_grass

        return self.board_state(), {"score": self.score}

    def board_state(self, mode="human", close=False):
        "Render the environment"
        self.board[:, :, :] = 0
        if self.max_grass > 0:
            self.board[:, :, 1] = self.grass / self.max_grass * 0.3
        if not self.done:
            x, y = self.snake[0]
            self.board[x, y, :] = 1
        for x, y in self.snake[1:]:
            self.board[x, y, 0] = 1
        for x, y in self.apples:
            self.board[x, y, 1] = 1
        if self.border == 0:
            return self.board
        else:
            h, w, _ = self.board.shape
            board = np.full(
                (h + self.border * 2, w + self.border * 2, 3), 0.5, np.float32
            )
            board[self.border : -self.border, self.border : -self.border] = self.board
            return board

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


def make_env(**kwars):
    def _init():
        return SnakeGame(**kwars)

    return _init







def closest_heuristic(state):
    score, apple, head, tail, direction = state

    # print("current dir", int_to_dir[direction])
    
    apple_y, apple_x = apple[0]
    head_y,head_x = head


    #bigger_x_diff = abs(apple_x - head_x) > abs(apple_y - head_y)
    bigger_x_diff = abs(apple_x - head_x) > 0

    # print("bigger x diff", bigger_x_diff)

    if bigger_x_diff:
        
        goal_dir = "right" if apple_x > head_x else "left"

    else:
        goal_dir = "down" if apple_y > head_y else "up"
        
        
    # print("goal dir", goal_dir)
    goal_dir = dir_to_int[goal_dir]

    diff = goal_dir - direction

    if abs(diff) == 0:
        action = 0

    elif abs(diff) == 1:
        action = 1 if goal_dir > direction else -1
        
    elif abs(diff) == 2: # chose randomly
        
        action = 1 if np.random.uniform(0,1)>0.5 else -1

    elif abs(diff) == 3: # go the other way

        action = -1 if goal_dir > direction else 1
        
    # print(action)
    

    return action



def heuristic_demo(heuristic):
    game = SnakeGame(30, 30, border=1, render_mode="human")
    board, _ = game.reset()
    action_name = {-1: "Turn left", 0: "Straight ahead", 1: "Turn right"}
    game.render()
    
    while True:
        
        new_action = heuristic(game.state)
        # print("new action", new_action)
        board, reward, terminated, _, info = game.step(new_action)
        game.render()
        
        if terminated:
            break


if __name__ == "__main__":

    heuristic_demo(closest_heuristic)
