import matplotlib.pyplot as plt
from snake_game import SnakeGameEnv
import time


def snake_demo(n_actions):
    game = SnakeGameEnv(30,30,border=1, render_mode= "human")
    board,_ = game.reset()    
    action_name = {-1:'Turn left',0:'Straight ahead',1:'Turn right'}    
    game.render()
    for _ in range(n_actions):
        board,reward,terminated, _ ,info = game.step(        game.action_space.sample())
        game.render()

snake_demo(200)
    
