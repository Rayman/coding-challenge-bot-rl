#!/usr/bin/python3
# %%
import tensorrt
from snakes.bots.brammmieee.env import SnakeEnv
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, StopTrainingOnNoModelImprovement
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks

# ============================== # Creating the environment # ============================ #
# %%
env = SnakeEnv()
check_env(env)
env = Monitor(
    env=env, 
    filename=None,
    info_keywords=(),  # can be used for logging the parameters for each test run for instance
)
env = DummyVecEnv([lambda: env])
env = VecNormalize(
    venv=env,
    training=True, 
    norm_obs=True, 
    norm_reward=True, 
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99,
    epsilon=1e-8,
    norm_obs_keys=None,
)

# ====================================== # Training # ==================================== #
# %% Model
model_name = "11__reward_standstill_and_lose_penalty"
model = MaskablePPO(
    policy="MlpPolicy",
    env=env,
    tensorboard_log = "./logs/" + model_name,
)

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq = 10000,
    save_path = "./models/" + model_name,
    name_prefix = model_name,
    save_replay_buffer = False,
    save_vecnormalize = False,
    verbose = 0,
)
eval_callback = EvalCallback(
    eval_env = env,
    callback_on_new_best = None,
    callback_after_eval = None,
    n_eval_episodes = 15,
    eval_freq = 10000,
    log_path = None,
    best_model_save_path = "./models",
    deterministic = False,
    render = False,
    verbose = 0,
    warn = True,
)
callback_list = CallbackList([ #NOTE: can also pass list directly to learn
    checkpoint_callback, 
    eval_callback,
    ]) 

# %% Train model
model.learn(
    total_timesteps=5e7,
    callback=callback_list,
    log_interval=10, 
    tb_log_name=model_name, 
    reset_num_timesteps=False, 
    progress_bar=True
)

# ================================= # Loading  Model # ================================== #
# %% Name
import re
zip_name = "10_vec_env__16x16__only_pos_progress_added_4500000_steps"
model_name = re.sub(r"_\d+_steps$", "", zip_name)

# %%
model = MaskablePPO.load(f"/home/bramo/coding-challenge-snakes/models/{model_name}/{zip_name}")

# %% Set new env
model.set_env(env=env)

# ============================ # Debugging # ============================================= #
# %%
def print_info(info):
    fig, ax = plt.subplots()
    cax = ax.imshow(info['grid_observation'], cmap='coolwarm', interpolation='nearest')
    fig.colorbar(cax)

    content = f"done = {info['done']:.3f}\n"
    content += f"action = {info['action']}\n"
    print(info['reward'])
    for key, value in info['reward'].items():
        content += f"{key} reward = {value:3f}\n"

    plt.text(1.9, 0.5, content, transform=plt.gca().transAxes,
        horizontalalignment='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
    plt.show()

# %% 
from snakes.bots.brammmieee.env import SnakeEnv
import numpy as np
import tensorrt
from icecream import ic
import matplotlib.pyplot as plt

# %%
env = SnakeEnv(debug=True, save_info=True)

# %%
obs = env.reset() 

# %%
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
print_info(info)
if done:
    obs = env.reset()

# %% 
n_steps = 1000
for i in range(n_steps):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
# %%
