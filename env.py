from typing import List, Tuple

import numpy as np
import gym
# import gymnasium as gym

from ...bot import Bot
from ...bots.random import Random, is_on_grid, collides
from ...constants import Move, MOVES, MOVE_VALUE_TO_DIRECTION
from ...game import Game
from ...snake import Snake
from icecream import ic


class SnakeEnv(gym.Env):
    def __init__(self, render=False):
        self.printer = Printer()
        self.render = render

        self.size = (16, 16)
        self.observation_space = gym.spaces.Box(low=-2, high=1, shape=self.size, dtype=np.float32) # TODO: check notes on normalization in [0,1]
        self.action_space = gym.spaces.Discrete(4)
        self.game = None
    
    def reset(self):
        self.agents = {
            0: MockAgent,
            1: Random,
        }
        self.game = Game(agents=self.agents)
        return self.get_obs()

    def step(self, action):
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

        observation = self.get_obs()
        done = self.game.finished()
        info = self.get_info()

        return observation, reward, done, info

    def render(self):
        if self.render:
            self.printer.print(self.game)

    def get_info(self):
        return {}

    def action_masks(self) -> np.ndarray:
        player = next(s for s in self.game.snakes if s.id == 0)

        return np.array([is_on_grid(player[0] + direction, self.game.grid_size)
                         and not collides(player[0] + direction, self.game.snakes)
                         for move, direction in MOVE_VALUE_TO_DIRECTION.items()], dtype=bool)

    def get_obs(self):
        player = next((s for s in self.game.snakes if s.id == 0), None)
        opponent = next((s for s in self.game.snakes if s.id == 0), None)
        return get_obs(self.game.grid_size, player, opponent, self.game.candies)


def get_obs(grid_size, player: Snake, opponent: Snake, candies: List[np.array]):
    grid = np.zeros(grid_size, dtype=np.float32)

    for candy in candies:
        grid[candy[0], candy[1]] = 1

    if player:
        for segment in player:
            grid[segment[0], segment[1]] = -2
        grid[player[0][0], player[0][1]] = -1
    if opponent:
        for segment in opponent:
            grid[segment[0], segment[1]] = -2

    return grid


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
