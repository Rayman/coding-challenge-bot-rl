import os.path
from typing import List, Tuple

import numpy as np
from sb3_contrib import MaskablePPO

from .env import get_obs
from ...bot import Bot
from ...bots.random import is_on_grid, collides
from ...constants import MOVE_VALUE_TO_DIRECTION
from ...constants import Move, MOVES
from ...snake import Snake


class RLQuaza(Bot):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        self.id = id
        self.grid_size = grid_size
        # zip_name = "14__teach_me_sensei__multibin_obs__higher_pos_reward_10480000_steps"
        # self.model = MaskablePPO.load(os.path.join(os.getcwd(), "snakes", "bots", "brammmieee", "models", zip_name))
        # self.model = MaskablePPO.load(os.path.join(os.getcwd(), "models", "best_model.zip"))
        self.model = MaskablePPO.load(
            os.path.join(os.getcwd(), 'models/v1_candy_distance/v1_candy_distance_5530000_steps.zip'))
        self.model = MaskablePPO.load(
            os.path.join(os.getcwd(), 'models/v1_small_policy/v1_small_policy_310000_steps.zip'))

    @property
    def name(self):
        return 'RLQuaza'

    @property
    def contributor(self):
        return 'brammmieee'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        opponent = other_snakes[0]
        observation = get_obs(self.grid_size, snake, opponent, candies)
        action, _ = self.model.predict(
            observation=observation,
            deterministic=True,
            action_masks=self.action_masks(snake, other_snakes)
        )
        return MOVES[action]

    def action_masks(self, player: Snake, other_snakes: List[Snake],
                     grid_size: Tuple[int, int] = (16, 16)) -> np.ndarray:
        return np.array([is_on_grid(player[0] + direction, grid_size)
                         and not collides(player[0] + direction, [player, other_snakes])
                         for move, direction in MOVE_VALUE_TO_DIRECTION.items()], dtype=bool)
