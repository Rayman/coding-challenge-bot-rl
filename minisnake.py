import gymnasium as gym
import numpy as np
import pygame


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
        self.window_size = np.array([800, 600])

        self.grid_size = np.array([8, 16])
        self.player = np.array([0, 0])
        self.candy = np.array([0, 0])

        self.observation_space = gym.spaces.Dict(
            {
                "candy_direction": gym.spaces.Box(np.array([-16, -16]), np.array([16, 16]), dtype=int),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

    def _get_obs(self):
        return {
            "candy_direction": self.candy - self.player
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
    id="minisnake-v0",
    entry_point="snakes.bots.brammmieee.minisnake:SnakeEnv",
    max_episode_steps=300,
)
