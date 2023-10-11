import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, Env
from gymnasium.envs.registration import register
from snakes.bot import Bot
from snakes.bots.random import Random
from snakes.game import Game


class SnakeEnv(Env):
    metadata = {'render_modes': ['ansii']}

    def __init__(self):
        self.size = (16, 16)

        self.observation_space = spaces.Dict(
            {
                'agent': spaces.Box(low=np.array([0, 0]), high=np.array(self.size), dtype=np.float32)
            }
        )

        self.action_space = spaces.Discrete(4)

        self.game = None

    def step(self, action: ActType):
        print('step')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print('reset!!!')

        agents = {
            0: MockAgent,
            1: Random,
        }
        self.game = Game(agents=agents)


class MockAgent(Bot):
    pass


register(
    id="snakes",
    entry_point="snakes.bots.brammmieee.env:SnakeEnv",
)
