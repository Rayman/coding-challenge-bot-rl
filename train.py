#!/usr/bin/python3
import os

from sb3_contrib import MaskablePPO
from snakes.bots.brammmieee.env import SnakeEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

env = SnakeEnv()
check_env(env)
# env = AsyncVectorEnv([lambda: SnakeEnv()])
env = Monitor(env=env)

# env = DummyVecEnv([lambda: env])


model_name = "v1_candy_distance"
model = MaskablePPO(
    policy="MultiInputPolicy",
    env=env,
    tensorboard_log=os.path.join("logs", model_name)
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/" + model_name,
    name_prefix=model_name,
    verbose=1,
)
eval_callback = EvalCallback(
    eval_env=env,
    n_eval_episodes=10,
    eval_freq=10000,
    best_model_save_path="./models",
    deterministic=False,
    verbose=1,
    warn=True,
)
callback_list = CallbackList([  # NOTE: can also pass list directly to learn
    checkpoint_callback,
    eval_callback,
])

# %% Train model
model.learn(
    total_timesteps=5e7,
    callback=callback_list,
    tb_log_name=model_name,
    reset_num_timesteps=False,
    progress_bar=True
)
