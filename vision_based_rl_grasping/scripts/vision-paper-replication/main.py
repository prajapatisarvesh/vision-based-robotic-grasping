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

from stable_baselines3 import PPO,DQN,A2C,DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

class ContinuousDownwardBiasPolicy(object):
  """Policy which takes continuous actions, and is biased to move down.
  """

  def __init__(self, height_hack_prob=0.9):
    """Initializes the DownwardBiasPolicy.

    Args:
        height_hack_prob: The probability of moving down at every move.
    """
    self._height_hack_prob = height_hack_prob
    self._action_space = spaces.Box(low=-1, high=1, shape=(5,))

  def sample_action(self, obs, explore_prob):
    """Implements height hack and grasping threshold hack.
    """
    dx, dy, dz, da, close = self._action_space.sample()
    if np.random.random() < self._height_hack_prob:
      dz = -1
    return [dx, dy, dz, da, 0]

def make_env(env_id: str, rank: int, seed: int=0):
  def _init():
    register_env(id='Kuka-v0', entry_point='pybullet_envs.bullet.kuka_diverse_object_gym_env:KukaDiverseObjectEnv')
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
      env.render(mode='human')
      act = policy.sample_action(obs, .1)
      print("Action")
      print(act)
      obs, rew, done, _ = env.step([0, 0, 0, 0, 0])
      episode_rew += rew
    print("Episode reward", episode_rew)

def differentPolicies(policy):
  register_env(id='Kuka-v0', entry_point='pybullet_envs.bullet.kuka_diverse_object_gym_env:KukaDiverseObjectEnv')
  # env = gym.make(id='Kuka-v0', renders=True, isDiscrete=False)
  # env = DummyVecEnv([lambda: env])
  env = SubprocVecEnv([make_env('Kuka-v0', i) for i in range(50)])
  model = policy("MlpPolicy",env,verbose=1, tensorboard_log='runs', n_steps = 1000)
  model.learn(total_timesteps=1e5, progress_bar=True)
  model.save('A2C-Kuka')
  print("PPO Policy Implementations")
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


if __name__ == '__main__':
  # main()
  differentPolicies(A2C)
