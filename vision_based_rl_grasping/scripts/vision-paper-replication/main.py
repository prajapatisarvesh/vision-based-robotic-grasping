"""Runs a random policy for the random object KukaDiverseObjectEnv.
"""

import os, inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from gym.envs import register as register_env
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces

from stable_baselines3 import PPO, DQN, A2C, DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

import os
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common import results_plotter


class ContinuousDownwardBiasPolicy(object):
    """Policy which takes continuous actions, and is biased to move down."""

    def __init__(self, height_hack_prob=0.9):
        """Initializes the DownwardBiasPolicy.

        Args:
            height_hack_prob: The probability of moving down at every move.
        """
        self._height_hack_prob = height_hack_prob
        self._action_space = spaces.Box(low=-1, high=1, shape=(5,))

    def sample_action(self, obs, explore_prob):
        """Implements height hack and grasping threshold hack."""
        dx, dy, dz, da, close = self._action_space.sample()
        if np.random.random() < self._height_hack_prob:
            dz = -1
        return [dx, dy, dz, da, 0]


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        print("Callback called")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        register_env(
            id="Kuka-v0",
            entry_point="pybullet_envs.bullet.kuka_diverse_object_gym_env:KukaDiverseObjectEnv",
        )
        env = gym.make(id=env_id, renders=False, isDiscrete=False)
        env.reset()
        return env

    set_random_seed(seed)
    return _init


def main():
    env = KukaDiverseObjectEnv(renders=True, isDiscrete=False)
    policy = ContinuousDownwardBiasPolicy()
    while True:
        obs, done = env.reset(), False
        print("===================================")
        print("obs")
        print(obs)
        episode_rew = 0
        while not done:
            env.render(mode="human")
            act = policy.sample_action(obs, 0.1)
            print("Action")
            print(act)
            obs, rew, done, _ = env.step([0, 0, 0, 0, 0])
            episode_rew += rew
        print("Episode reward", episode_rew)


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()


def differentPolicies(policy, Vectorized=False, nenv=1):
    # Create log dir
    log_dir = "/home/yash/Documents/vision-based-robotic-grasping/vision_based_rl_grasping/scripts/vision-paper-replication/logs/"
    os.makedirs(log_dir, exist_ok=True)
    print("Temporary log dir: {}".format(log_dir))

    register_env(
        id="Kuka-v0",
        entry_point="pybullet_envs.bullet.kuka_diverse_object_gym_env:KukaDiverseObjectEnv",
    )
    # env = gym.make(id='Kuka-v0', renders=True, isDiscrete=False)
    # env = DummyVecEnv([lambda: env])
    if Vectorized:
        nenv = 50
    else:
        nenv = 1
    env = SubprocVecEnv([make_env("Kuka-v0", i) for i in range(nenv)])
    # callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model = policy("MlpPolicy", env, verbose=1, tensorboard_log="runs", n_steps=1000)
    evalCallback = EvalCallback(
        env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=500,
    )
    model.learn(total_timesteps=1e5, progress_bar=True, callback=evalCallback)
    model.save("A2C-Kuka")
    print("PPO Policy Implementations")

    # Helper from the library
    # results_plotter.plot_results(
    #     [log_dir], 1e5, results_plotter.X_TIMESTEPS, "Kuka-v0"
    # )

    # plot_results(log_dir)
    env.close()

    # env = gym.make(id='Kuka-v0', renders=True, isDiscrete=False)
    # env = DummyVecEnv([lambda: env])
    # for _ in range(5):
    #   obs, done = env.reset(), False
    #   # print("===================================")
    #   # print("obs")
    #   # print(obs)
    #   episode_rew = 0
    #   while not done:
    #     # env.render(mode='human')
    #     act,_ = model.predict(obs)
    #     # print("Action")
    #     # print(act)
    #     obs, rew, done, _ = env.step(act)
    #     episode_rew += rew
    #   print("Episode reward", episode_rew)
    #   env.close()


if __name__ == "__main__":
    # main()
    differentPolicies(A2C, Vectorized=True, nenv=15)
