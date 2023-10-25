from typing import List, Tuple

import numpy as np
import gym

from ...bot import Bot
from ...bots.random import Random, is_on_grid, collides
from ...constants import Move, MOVES, MOVE_VALUE_TO_DIRECTION
from ...game import Game
from ...snake import Snake
from copy import deepcopy
import random

from ..example.bot import ExampleBot
from ..random import Random
from ..hein.bot import ApologeticApophis
from ..felipe.bot import TemplateSnake
from ..mahmoud.bot import SneakyBot
from ..jeroen.bot import ExampleBot as JeroenBot
from ..jonothan.bot import bender
from ..lewie.bot import LewieBot
from ..bram.bot import Slytherin
from ..daniel.bot import Explorer
from ..rokusottervanger.bot import OtterByte
from ..mukunda.bot import Snakunamatata
from ..ferry.bot import FurryMuncher
from ..mhoogesteger.bot import CherriesAreForLosers

class SnakeEnv(gym.Env):
    def __init__(self, render=False, debug=False):
        bots = (
            Random,
            ExampleBot,
            ApologeticApophis,
            TemplateSnake,
            SneakyBot,
            JeroenBot,
            # bender,
            LewieBot,
            Slytherin,
            Explorer,
            OtterByte,
            Snakunamatata,
            FurryMuncher,
            CherriesAreForLosers,
        )
        self.first_loop = True
        self.random_bot = bots[random.randint(0, len(bots)-1)]
        self.printer = Printer()
        self.render = render
        self.debug = debug

        self.size = (16, 16)
        self.observation_space = gym.spaces.Box(low=-4, high=1, shape=self.size, dtype=np.float32) # TODO: check notes on normalization in [0,1]
        self.action_space = gym.spaces.Discrete(4)
        self.game = None
        self.info = {}
    
    def reset(self):
        self.agents = {
            0: MockAgent,
            # 1: self.random_bot,
            1: Random,
        }
        self.game = Game(agents=self.agents, print_stats=self.debug)
        self.player = next((s for s in self.game.snakes if s.id == 0), None)
        self.opponent = next((s for s in self.game.snakes if s.id == 1), None)
        
        observation = self.get_obs(self.player, self.opponent)
        return observation

    def step(self, action):
        self.game.agents[0].next_move = MOVES[action]
        player_prev = deepcopy(self.player)
        opponent_prev = deepcopy(self.opponent)

        # update the game
        self.game.update()  # Let our bot play
        self.player = next((s for s in self.game.snakes if s.id == 0), None) # update bot pos
        self.game.update()  # Let the opponent play
        self.opponent = next((s for s in self.game.snakes if s.id == 1), None) # update opponent pos
        
        # When snake becomes None or is outside the bounds set to prev position for last state
        if not self.player:
            self.player = player_prev
        if not self.opponent:
            self.opponent = opponent_prev  

        # Get return values
        reward = self.get_reward(self.player)
        observation = self.get_obs(self.player, self.opponent)
        done = self.game.finished()

        self.info.update({"done": done})
        self.info.update({"action": self.game.agents[0].next_move})
        info = self.get_info()

        return observation, reward, done, info
    
    def get_reward(self, player: Snake):
        snake_head = np.array([player[0][0], player[0][1]])
        candies = self.game.candies

        if not self.first_loop:
            closest_candy_dist_prev = self.closest_candy_dist
        candy_dists = [np.linalg.norm(snake_head - candy) for candy in candies]
        self.closest_candy_dist = min(candy_dists)

        # finish reward
        if self.game.finished():
            if self.game.scores[0] >= self.game.scores[1]:
                finish_reward = 100
            else:
                finish_reward = -100
        else:
            finish_reward = 0

        # candy reward
        if any(np.array_equal(snake_head, candy) for candy in candies):
            candy_reward = 10
        else:
            candy_reward = 0
        
        # progress reward
        if not self.first_loop:
            progress_reward = 5*(closest_candy_dist_prev - self.closest_candy_dist)
        else:
            progress_reward = 0

        # total reward
        reward = finish_reward + candy_reward + progress_reward
        reward_dict = {
            "finish": finish_reward,
            "cany": candy_reward,
            "progress": progress_reward,
            "total": reward,
        }
        self.info.update({
            "reward": reward_dict
        })
        
        self.first_loop = False
        return reward        

    def action_masks(self) -> np.ndarray:
        player = next(s for s in self.game.snakes if s.id == 0)
        return np.array([is_on_grid(player[0] + direction, self.game.grid_size)
                         and not collides(player[0] + direction, self.game.snakes)
                         for move, direction in MOVE_VALUE_TO_DIRECTION.items()], dtype=bool)

    def get_obs(self, player: Snake, opponent: Snake):
        observation = get_obs(self.game.grid_size, player, opponent, self.game.candies)
        self.info.update({"observation": observation})
        return observation

    def render(self):
        if self.render:
            self.printer.print(self.game)

    def get_info(self):
        return self.info

def get_obs(grid_size, player: Snake, opponent: Snake, candies: List[np.array]):
    grid = np.zeros(grid_size, dtype=np.float32)

    # fill the grid (obstacles:1, candies:2)
    for segment in player:
        grid[segment[0], segment[1]] = 1
    for segment in opponent:
        grid[segment[0], segment[1]] = 1
    for candy in candies:
        grid[candy[0], candy[1]] = 2

    # compute local observation around snake head
    snake_head = np.array([player[0][0], player[0][1]])
    if player[0][0] > player[1][0]:
        snake_heading = "right" 
    elif player[0][0] < player[1][0]:
        snake_heading = "left"
    elif player[1][0] > player[1][1]:
        snake_heading = "up"
    elif player[1][0] < player[1][1]:
        snake_heading = "down"

    

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
