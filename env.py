import random
from typing import List, Tuple

import gymnasium as gym
import numpy as np

from ..random import Random
from ...bot import Bot
from ...bots.random import is_on_grid, collides
from ...constants import Move, MOVES, MOVE_VALUE_TO_DIRECTION
from ...game import Game
from ...snake import Snake


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        bots = (
            Random,
        )
        self.first_loop = True
        self.random_bot = bots[random.randint(0, len(bots) - 1)]
        self.printer = Printer()

        grid_length = 16
        self.size = (grid_length, grid_length)
        # self.observation_space = gym.spaces.MultiBinary((grid_length ** 2) * 4)
        self.observation_space = gym.spaces.Dict({
            # 'candies': gym.spaces.MultiBinary((16, 16)),
            # 'player': gym.spaces.MultiBinary((16, 16)),
            # 'opponent': gym.spaces.MultiBinary((16, 16)),
            # 'occupied': gym.spaces.MultiBinary((16, 16)),
            'candy_vectors': gym.spaces.Box(-16, 16, (3, 2), dtype=np.int8)
        })
        self.action_space = gym.spaces.Discrete(4)
        self.game = None
        self.player = None
        self.opponent = None

    def reset(self, seed=None):
        # print('reset after turns', self.game.turns if self.game else None)
        super().reset(seed=seed)
        agents = {
            0: MockAgent,
            1: self.random_bot,
        }
        self.game = Game(grid_size=self.size, agents=agents)
        self.player = next((s for s in self.game.snakes if s.id == 0), None)
        self.opponent = next((s for s in self.game.snakes if s.id == 1), None)

        observation = self.get_obs(self.player, self.opponent)
        return observation, {}

    def step(self, action):
        action_move = MOVES[action]
        self.game.agents[0].next_move = action_move

        length_before = len(self.player)
        candy_distance_before = min([np.linalg.norm(self.player[0] - c, 1) for c in self.game.candies])

        # update the game
        self.game.update()  # Let our bot play
        if self.player in self.game.snakes:
            self.game.update()  # Let the opponent play if our bot doesn't die first (other bot needs our bot to update it's move)

        terminated = self.game.finished()
        truncated = False

        reward = 0
        if terminated:
            if self.game.rank()[0] < self.game.rank()[1]:
                reward += 10
            elif self.game.rank()[0] > self.game.rank()[1]:
                reward -= 10

        if self.player not in self.game.snakes:
            self.player = None
        else:
            length_difference = len(self.player) - length_before
            reward += length_difference

            # candy_distance_difference = min(
            #     [np.linalg.norm(self.player[0] - c, 1) for c in self.game.candies]) - candy_distance_before
            # reward += -candy_distance_difference / 16
        if self.opponent not in self.game.snakes:
            self.opponent = None
        observation = self.get_obs(self.player, self.opponent)

        return observation, reward, terminated, truncated, {}

    def action_masks(self) -> np.ndarray:
        player = next(s for s in self.game.snakes if s.id == 0)
        return np.array([is_on_grid(player[0] + direction, self.game.grid_size)
                         and not collides(player[0] + direction, self.game.snakes)
                         for move, direction in MOVE_VALUE_TO_DIRECTION.items()], dtype=bool)

    def get_obs(self, player: Snake, opponent: Snake):
        observation = get_obs(self.game.grid_size, player, opponent, self.game.candies)
        return observation

    def render(self):
        self.printer.print(self.game)


def get_obs(grid_size, player: Snake, opponent: Snake, candies: List[np.array]):
    candy_grid = np.zeros(grid_size, dtype=np.int8)
    player_head_grid = np.zeros(grid_size, dtype=np.int8)
    opponent_head_grid = np.zeros(grid_size, dtype=np.int8)
    occupied_grid = np.zeros(grid_size, dtype=np.int8)

    candy_vectors = np.zeros((3, 2), dtype=np.int8)
    for candy in candies:
        candy_grid[candy[0], candy[1]] = 1

    if player:
        player_head_grid[player[0][0], player[0][1]] = 1
        for segment in player:
            occupied_grid[segment[0], segment[1]] = 1

        for i, candy in enumerate(candies):
            candy_vectors[i] = player[0] - candy

    if opponent:
        opponent_head_grid[opponent[0][0], opponent[0][1]] = 1
        for segment in opponent:
            occupied_grid[segment[0], segment[1]] = 1

    observation = {
        # 'candies': candy_grid,
        # 'player': player_head_grid,
        # 'opponent': opponent_head_grid,
        # 'occupied': occupied_grid,
        'candy_vectors': candy_vectors,
    }
    return observation

    # observation = np.concatenate((candy_grid, player_head_grid, opponent_head_grid, occupied_grid))
    # return observation


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
