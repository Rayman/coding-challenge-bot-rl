from typing import List, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, Env
from gymnasium.envs.registration import register

from ...bot import Bot
from ...bots.random import Random
from ...constants import Move, MOVES
from ...game import Game
from ...snake import Snake


class SnakeEnv(Env):
    metadata = {'render_modes': ['ansii']}

    def __init__(self):
        self.size = (16, 16)

        self.observation_space = spaces.Dict(
            {
                'grid': spaces.Box(low=0, high=1, shape=self.size, dtype=np.float32)
            }
        )

        self.action_space = spaces.Discrete(4)

        self.game = None

    def step(self, action: ActType):
        self.game.agents[0].next_move = MOVES[action]
        self.game.update()

        if self.game.finished():
            reward = self.game.scores[0]
        else:
            player = next(s for s in self.game.snakes if s.id == 0)
            opponent = next(s for s in self.game.snakes if s.id == 1)
            reward = len(player)

        observation = self._get_obs()
        terminated = self.game.finished()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        agents = {
            0: MockAgent,
            1: Random,
        }
        self.game = Game(agents=agents)

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        player = (s for s in self.game.snakes if s.id == 0)

        grid = np.zeros(self.game.grid_size, dtype=np.float32)
        for snake in self.game.snakes:
            for segment in snake:
                grid[segment] = 1

        return {'grid': grid}

    def _get_info(self):
        return {}


class MockAgent(Bot):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        self.grid_size = grid_size
        self.next_move = None

    @property
    def name(self):
        return 'MockAgent'

    @property
    def contributor(self):
        return 'brammmieee'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        return self.next_move


register(
    id="snakes",
    entry_point="snakes.bots.brammmieee.env:SnakeEnv",
)
