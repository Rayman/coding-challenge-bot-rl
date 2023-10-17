# %% Modules
from snakes.bots.brammmieee.env import SnakeEnv
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy
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

# ====================================== # Training # ==================================== #
# %% Model #NOTE: alternatively run the "Loading non-archived model section!"
model_name = "2_action_masked"
model = MaskablePPO(
    policy="MlpPolicy",
    env=env,
    tensorboard_log = "./logs/" + model_name,
)

# %% Callbacks
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
    n_eval_episodes = 10,
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
    total_timesteps=2e6,
    callback=callback_list,
    log_interval=10, 
    tb_log_name=model_name, 
    reset_num_timesteps=False, 
    progress_bar=True
)

# ================================= # Loading  Model # ================================== #
# %% Name
model_name = 'juli20V1_wp_progr_reward'
n_steps_load = 220000

# %% Load model
model = PPO.load('./models' + '/' + model_name +  '/' + model_name + '_' + str(n_steps_load) + '_steps')

# %% Set new env
model.set_env(env=env)

# =================================== # Evaluating policy # ============================= #
# %% 
env = SnakeEnv()

# %% Load model
model = PPO.load('./models' + '/')

# %% 
nr_eval_eps = 10

# %% Evaluation runs
mean_reward, std_reward = evaluate_policy(
    model, 
    env=env, 
    n_eval_episodes=nr_eval_eps, 
    deterministic=True, 
    render=False,
    callback=None,
    reward_threshold=None,
    return_episode_rewards=False,
    warn=True,
)
print(f'mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}')

# ============================ # Debugging # ============================================= #
# %% 
from snakes.bots.brammmieee.env import SnakeEnv
import numpy as np
from icecream import ic

# %%
env = SnakeEnv()

# %%
obs = env.reset() 

# %%
action = env.action_space.sample()
obs, reward, done, info = env.step(action)

# %% 
import time

# obs = env.reset()
n_steps = 100000000
for i in range(n_steps):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action, teleop=True)
    # time.sleep(0.4)
    if done:
        obs = env.reset()
# %%
