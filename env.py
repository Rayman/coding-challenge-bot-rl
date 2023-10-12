import random
from typing import List, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, Env
from gymnasium.envs.registration import register

from ...bot import Bot
from ...bots.random import Random, is_on_grid, collides
from ...constants import Move, MOVES, MOVE_VALUE_TO_DIRECTION
from ...game import Game
from ...snake import Snake


class SnakeEnv(Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        self.size = (16, 16)

        self.observation_space = spaces.Dict(
            {
                'grid': spaces.Box(low=-2, high=1, shape=self.size, dtype=np.float32)
            }
        )

        self.action_space = spaces.Discrete(4)

        self.game = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def step(self, action: ActType):
        player = next(s for s in self.game.snakes if s.id == 0)
        length_before_update = len(player)

        self.game.agents[0].next_move = MOVES[action]
        self.game.update()  # Let our bot play
        self.game.update()  # Let the opponent play

        if self.game.finished():
            if self.game.scores[0] >= self.game.scores[1]:
                reward = 10
            else:
                reward = -10
            # print('final reward:', reward)
        else:
            # player = next(s for s in self.game.snakes if s.id == 0)
            # opponent = next(s for s in self.game.snakes if s.id == 1)
            reward = len(player) - length_before_update
            # print('reward:', reward)

        observation = self._get_obs()
        terminated = self.game.finished()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        agents = {
            0: MockAgent,
            1: Random,
        }
        self.game = Game(agents=agents)

        return self._get_obs(), self._get_info()

    def render(self):
        printer = Printer()
        printer.print(self.game)

    def action_masks(self) -> np.ndarray:
        player = next(s for s in self.game.snakes if s.id == 0)

        return np.array([is_on_grid(player[0] + direction, self.game.grid_size)
                         and not collides(player[0] + direction, self.game.snakes)
                         for move, direction in MOVE_VALUE_TO_DIRECTION.items()], dtype=bool)

    def _get_obs(self):

        grid = np.zeros(self.game.grid_size, dtype=np.float32)

        for candy in self.game.candies:
            grid[candy[0], candy[1]] = 1

        for snake in self.game.snakes:
            for segment in snake:
                grid[segment[0], segment[1]] = -2

        player = next((s for s in self.game.snakes if s.id == 0), None)
        if player:
            grid[player[0][0], player[0][1]] = -1

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


numbers = ['⓪']


def fill_numbers():
    one = '①'.encode()
    for i in range(20):
        ba = bytearray(one)
        ba[2] += i
        numbers.append(bytes(ba).decode())


fill_numbers()


def number_to_circled(number: int) -> str:
    return numbers[number % len(numbers)]


class Printer:
    def print(self, game):
        grid = np.empty(game.grid_size, dtype=str)
        grid.fill(' ')
        for candy in game.candies:
            grid[candy[0], candy[1]] = '*'
        for snake in game.snakes:
            print(f'name={game.agents[snake.id].name!r} {snake}')
            for pos in snake:
                grid[pos[0], pos[1]] = number_to_circled(snake.id)

        print(f' {"▁" * 2 * game.grid_size[0]}▁ ')
        for j in reversed(range(grid.shape[1])):
            print('▕', end='')
            for i in range(grid.shape[0]):
                print(f' {grid[i, j]}', end='')
            print(' ▏')
        print(f' {"▔" * 2 * game.grid_size[0]}▔ ')


register(
    id="snakes",
    entry_point="snakes.bots.brammmieee.env:SnakeEnv",
)
