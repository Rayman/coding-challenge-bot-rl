import os.path
from typing import List, Tuple

import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
import re

from .env import SnakeEnv, get_obs
from ...bots.random import Random, is_on_grid, collides
from ...constants import Move, MOVES, MOVE_VALUE_TO_DIRECTION
from ...bot import Bot
from ...constants import Move, MOVES
from ...snake import Snake


class RLQuaza(Bot):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        self.id = id
        self.grid_size = grid_size
        zip_name = "11__reward_standstill_and_lose_penalty_9000000_steps"
        model_name = re.sub(r"_\d+_steps$", "", zip_name)
        self.model = MaskablePPO.load(f"/home/bramo/coding-challenge-snakes/models/{model_name}/{zip_name}")

    @property
    def name(self):
        return 'RLQuaza'

    @property
    def contributor(self):
        return 'brammmieee'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        opponent = other_snakes[0]
        obs, _ = get_obs(self.grid_size, snake, opponent, candies)
        action, _ = self.model.predict(
            obs, 
            deterministic=True, 
            action_masks=self.action_masks(snake, other_snakes)
            )
        print(action)
        return MOVES[action]

    def action_masks(self, player: Snake, other_snakes: List[Snake], grid_size: Tuple[int, int] = (16, 16)) -> np.ndarray:
        return np.array([is_on_grid(player[0] + direction, grid_size)
                         and not collides(player[0] + direction, [player, other_snakes])
                         for move, direction in MOVE_VALUE_TO_DIRECTION.items()], dtype=bool)