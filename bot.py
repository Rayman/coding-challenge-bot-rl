import os.path
from typing import List, Tuple

import numpy as np
from stable_baselines3 import PPO

from .env import SnakeEnv, get_obs
from ...bot import Bot
from ...constants import Move, MOVES
from ...snake import Snake


class RLQuaza(Bot):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        self.id = id
        self.grid_size = grid_size

        # %% Load model
        self.model = PPO.load(os.path.join(os.path.dirname(__file__), 'models/best_model'))

        # %% Set new env
        env = SnakeEnv()
        self.model.set_env(env=env)

    @property
    def name(self):
        return 'RLQuaza'

    @property
    def contributor(self):
        return 'brammmieee'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        opponent = other_snakes[0]
        obs = get_obs(self.grid_size, snake, opponent, candies)
        action, _ = self.model.predict(obs, deterministic=True)
        print(action)
        return MOVES[action]
