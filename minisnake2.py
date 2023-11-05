import gymnasium as gym
import numpy as np
import pygame
import torch as th
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import CombinedExtractor, BaseFeaturesExtractor
from stable_baselines3.ppo import MultiInputPolicy
from torch import nn


class CustomNatureCNN(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 512,
            normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, gym.spaces.Box), (
            "CustomNatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use CustomNatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            # nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            # nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        print(self)
        raise

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))



CombinedExtractor
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            cnn_output_dim: int = 16,
            normalized_image: bool = False):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)



        super().__init__(observation_space, cnn_output_dim, normalized_image)
        subspace = observation_space.spaces['candy_grid']
        self.extractors['candy_grid'] = CustomNatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=True)


class CustomMultiInputPolicy(MultiInputPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=CustomCombinedExtractor)


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    _action_to_direction = {
        0: np.array([1, 0]),
        1: np.array([0, 1]),
        2: np.array([-1, 0]),
        3: np.array([0, -1]),
    }

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = np.array([800, 800])

        self.grid_size = np.array([16, 16])
        self.player = np.array([0, 0])
        self.candy = np.array([0, 0])

        self.observation_space = gym.spaces.Dict(
            {
                # "candy_direction": gym.spaces.Box(np.array([-16, -16]), np.array([16, 16]), dtype=int),
                "player_grid": gym.spaces.Box(0, 255, (1, *self.grid_size), dtype=np.uint8),
                "candy_grid": gym.spaces.Box(0, 255, (1, *self.grid_size), dtype=np.uint8),
            }
        )
        assert is_image_space(self.observation_space['player_grid'])
        assert is_image_space(self.observation_space['candy_grid'])

        self.action_space = gym.spaces.Discrete(4)

    def _get_obs(self):
        player_grid = np.zeros(self.grid_size, dtype=bool)
        candy_grid = np.zeros(self.grid_size, dtype=bool)
        player_grid[tuple(self.player)] = 1
        candy_grid[tuple(self.candy)] = 1
        return {
            # "candy_direction": self.candy - self.player,
            "player_grid": player_grid,
            "candy_grid": candy_grid,
        }

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player = np.random.randint(self.grid_size, size=(2,))
        self.candy = np.random.randint(self.grid_size, size=(2,))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self.player = np.clip(
            self.player + direction, (0, 0), self.grid_size - np.array(1)
        )

        terminated = np.array_equal(self.player, self.candy)
        reward = 1 if terminated else 0  # Binary sparse rewards

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.grid_size

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 255, 0),
            pygame.Rect(self.candy * pix_square_size, pix_square_size),
        )
        # Now we draw the agent
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(self.player * pix_square_size, pix_square_size),
        )

        # Finally, add some gridlines
        for y in range(self.grid_size[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size[1] * y),
                (self.window_size[0], pix_square_size[1] * y),
                width=3,
            )
        for x in range(self.grid_size[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size[0] * x, 0),
                (pix_square_size[0] * x, self.window_size[1]),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


gym.register(
    id="minisnake-v2",
    entry_point="snakes.bots.brammmieee.minisnake2:SnakeEnv",
    max_episode_steps=300,
)
