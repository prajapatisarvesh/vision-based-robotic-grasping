import matplotlib.pyplot as plt
import sys
from collections import deque
import timeit
from datetime import timedelta
from copy import deepcopy
import numpy as np
import random
from PIL import Image
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import pybullet as p

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import argparse

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import collections
from itertools import count
import timeit
from datetime import timedelta
from PIL import Image
import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import pybullet as p
import argparse


import matplotlib.pyplot as plt
import sys
from collections import deque
import timeit
from datetime import timedelta
from copy import deepcopy
import numpy as np
import random
from PIL import Image
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import pybullet as p

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import argparse


import matplotlib.pyplot as plt
import sys
from collections import deque
import timeit
from datetime import timedelta
from copy import deepcopy
import numpy as np
import random
from PIL import Image
from tensorboardX import SummaryWriter

import functools
import multiprocessing as mp
from multiprocessing import Pipe
from multiprocessing import Process
import signal
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim

from gym import spaces
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import pybullet as p

import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import argparse


# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-e", "--EVAL", help="If want to evaluate the model or not", default=False
)

parser.add_argument(
    "p", "--policy", help="Whcih policy to train", default=False
)

args = parser.parse_args()

def hiddenLayers(input_dim, hidden_layers):
    hidden = nn.ModuleList([nn.Linear(input_dim, hidden_layers[0])])

    if len(hidden_layers) > 1:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        hidden.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

    return hidden


class A2C(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        shared_layers,
        critic_hidden_layers=[],
        actor_hidden_layers=[],
        seed=0,
    ):
        super(A2C, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.sigma = nn.Parameter(torch.zeros(action_size))

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[0])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[1])))
        linear_input_size = convh * convw * 32
        self.shared_layers = hiddenLayers(
            input_dim=linear_input_size, hidden_layers=shared_layers
        )

        if critic_hidden_layers:
            self.critic_hidden = hiddenLayers(
                input_dim=shared_layers[-1], hidden_layers=critic_hidden_layers
            )
            self.critic = nn.Linear(critic_hidden_layers[-1], 1)
        else:
            self.critic_hidden = None
            self.critic = nn.Linear(shared_layers[-1], 1)

        if actor_hidden_layers:
            self.actor_hidden = hiddenLayers(
                input_dim=shared_layers[-1], hidden_layers=actor_hidden_layers
            )
            self.actor = nn.Linear(actor_hidden_layers[-1], action_size)
        else:
            self.actor_hidden = None
            self.actor = nn.Linear(shared_layers[-1], action_size)

        self.tanh = nn.Tanh()

        if self.init_type is not None:
            self.shared_layers.apply(self._initialize)
            self.critic.apply(self._initialize)
            self.actor.apply(self._initialize)
            if self.critic_hidden is not None:
                self.critic_hidden.apply(self._initialize)
            if self.actor_hidden is not None:
                self.actor_hidden.apply(self._initialize)

    def _initialize(self, n):
        if isinstance(n, nn.Linear):
            nn.init.uniform_(n.weight.data)

    def forward(self, state):
        def apply_multi_layer(layers, x, f=F.leaky_relu):
            for layer in layers:
                x = f(layer(x))
            return x

        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.bn3(self.conv3(state)))
        state = apply_multi_layer(self.shared_layers, state.view(state.size(0), -1))

        v_hid = state
        if self.critic_hidden is not None:
            v_hid = apply_multi_layer(self.critic_hidden, v_hid)

        a_hid = state
        if self.actor_hidden is not None:
            a_hid = apply_multi_layer(self.actor_hidden, a_hid)

        a = self.tanh(self.actor(a_hid))
        value = self.critic(v_hid).squeeze(-1)
        return a, value


def get_screen():
    screen = env._get_observation().transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)


def collect_trajectories(envs, policy, tmax=200, nrand=5):
    global i_episode
    global ten_rewards
    global writer

    state_list = []
    reward_list = []
    prob_list = []
    action_list = []
    value_list = []
    done_list = []

    state = envs.reset()

    for _ in range(nrand):
        action = np.random.randn(action_size)
        action = np.clip(action, -1.0, 1.0)
        _, reward, done, _ = envs.step(action)
        reward = torch.tensor([reward], device=device)

    for t in range(tmax):
        states = get_screen()
        action_est, values = policy(states)
        sigma = nn.Parameter(torch.zeros(action_size))
        dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=-1).detach()
        values = values.detach()
        actions = actions.detach()

        env_actions = actions.cpu().numpy()
        _, reward, done, _ = envs.step(env_actions[0])
        rewards = torch.tensor([reward], device=device)
        dones = torch.tensor([done], device=device)

        state_list.append(states.unsqueeze(0))
        prob_list.append(log_probs.unsqueeze(0))
        action_list.append(actions.unsqueeze(0))
        reward_list.append(rewards.unsqueeze(0))
        value_list.append(values.unsqueeze(0))
        done_list.append(dones)

        if np.any(dones.cpu().numpy()):
            ten_rewards += reward
            i_episode += 1
            state = envs.reset()
            if i_episode % 10 == 0:
                writer.add_scalar(
                    "ten episodes average rewards", ten_rewards / 10.0, i_episode
                )
                ten_rewards = 0

    state_list = torch.cat(state_list, dim=0)
    prob_list = torch.cat(prob_list, dim=0)
    action_list = torch.cat(action_list, dim=0)
    reward_list = torch.cat(reward_list, dim=0)
    value_list = torch.cat(value_list, dim=0)
    done_list = torch.cat(done_list, dim=0)
    return prob_list, state_list, action_list, reward_list, value_list, done_list


def calc_returns(rewards, values, dones):
    n_step = len(rewards)
    n_agent = len(rewards[0])

    GAE = torch.zeros(n_step, n_agent).float().to(device)
    returns = torch.zeros(n_step, n_agent).float().to(device)

    GAE_current = torch.zeros(n_agent).float().to(device)

    TAU = 0.8
    discount = 0.99
    values_next = values[-1].detach()
    returns_current = values[-1].detach()
    for irow in reversed(range(n_step)):
        values_current = values[irow]
        rewards_current = rewards[irow]
        gamma = discount * (1.0 - dones[irow].float())

        td_error = rewards_current + gamma * values_next - values_current
        GAE_current = td_error + gamma * TAU * GAE_current
        returns_current = rewards_current + gamma * returns_current
        GAE[irow] = GAE_current
        returns[irow] = returns_current

        values_next = values_current

    return GAE, returns


def eval_policy(envs, policy, tmax=1000):
    reward_list = []
    state = envs.reset()
    for t in range(tmax):
        states = get_screen()
        action_est, values = policy(states)
        sigma = nn.Parameter(torch.zeros(action_size))
        dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))
        actions = dist.sample()
        _, reward, done, _ = envs.step(actions[0])
        dones = done
        reward_list.append(np.mean(reward))
        if np.any(dones):
            break
    return reward_list


def worker(remote, env_fn):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == "get_screen":
                screen = env._get_observation()
                screen = env._get_observation().transpose((2, 0, 1))
                screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
                screen = torch.from_numpy(screen)
                screen = resize(screen).unsqueeze(0)
                remote.send(screen)
            elif cmd == "reset":
                ob = env.reset()
                remote.send(ob)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.action_space, env.observation_space))
            else:
                raise NotImplementedError
    finally:
        env.close()


class VecEnv:
    def __init__(self, env_fns):
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, env_fn))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)
        ]
        for p in self.ps:
            p.start()
        self.last_obs = [None] * self.num_envs
        self.remotes[0].send(("get_spaces", None))
        self.action_space, self.observation_space = self.remotes[0].recv()
        self.closed = False

    def __del__(self):
        if not self.closed:
            self.close()

    def step(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        self.last_obs, rews, dones, infos = zip(*results)
        return self.last_obs, rews, dones, infos

    def get_screen(self):
        for remote in self.remotes:
            remote.send(("get_screen", None))
        results = [remote.recv() for remote in self.remotes]
        screens = torch.cat(results, dim=0)
        return screens

    def reset(self, mask=None):
        self._assert_not_closed()
        if mask is None:
            mask = np.zeros(self.num_envs)
        for m, remote in zip(mask, self.remotes):
            if not m:
                remote.send(("reset", None))

        obs = [
            remote.recv() if not m else o
            for m, remote, o in zip(mask, self.remotes, self.last_obs)
        ]
        self.last_obs = obs
        return obs

    def close(self):
        self._assert_not_closed()
        self.closed = True
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)

    def _assert_not_closed(self):
        assert not self.closed, "This env is already closed"


def make_env(idx, test):
    env = KukaDiverseObjectEnv(
        renders=False, isDiscrete=False, removeHeightHack=False, maxSteps=20
    )
    env.observation_space = spaces.Box(
        low=0.0, high=1.0, shape=(84, 84, 3), dtype=np.float32
    )
    env.action_space = spaces.Box(low=-1, high=1, shape=(5, 1))
    return env


def make_batch_env(test):
    return VecEnv(
        [functools.partial(make_env, idx, test) for idx in range(mp.cpu_count() * 2)]
    )


def hiddenLayers(input_dim, hidden_layers):
    hidden = nn.ModuleList([nn.Linear(input_dim, hidden_layers[0])])
    if len(hidden_layers) > 1:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        hidden.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    return hidden


class A2C(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        shared_layers,
        critic_hidden_layers=[],
        actor_hidden_layers=[],
        seed=0,
    ):
        super(A2C, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.sigma = nn.Parameter(torch.zeros(action_size))

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[0])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[1])))
        linear_input_size = convh * convw * 32
        self.shared_layers = hiddenLayers(
            input_dim=linear_input_size, hidden_layers=shared_layers
        )

        if critic_hidden_layers:
            self.critic_hidden = hiddenLayers(
                input_dim=shared_layers[-1], hidden_layers=critic_hidden_layers
            )
            self.critic = nn.Linear(critic_hidden_layers[-1], 1)
        else:
            self.critic_hidden = None
            self.critic = nn.Linear(shared_layers[-1], 1)

        if actor_hidden_layers:
            self.actor_hidden = hiddenLayers(
                input_dim=shared_layers[-1], hidden_layers=actor_hidden_layers
            )
            self.actor = nn.Linear(actor_hidden_layers[-1], action_size)
        else:
            self.actor_hidden = None
            self.actor = nn.Linear(shared_layers[-1], action_size)

        self.tanh = nn.Tanh()

        if self.init_type is not None:
            self.shared_layers.apply(self._initialize)
            self.critic.apply(self._initialize)
            self.actor.apply(self._initialize)
            if self.critic_hidden is not None:
                self.critic_hidden.apply(self._initialize)
            if self.actor_hidden is not None:
                self.actor_hidden.apply(self._initialize)

    def _initialize(self, n):
        if isinstance(n, nn.Linear):
            nn.init.uniform_(n.weight.data)

    def forward(self, state):
        def apply_multi_layer(layers, x, f=F.leaky_relu):
            for layer in layers:
                x = f(layer(x))
            return x

        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.bn3(self.conv3(state)))
        state = apply_multi_layer(self.shared_layers, state.view(state.size(0), -1))

        v_hid = state
        if self.critic_hidden is not None:
            v_hid = apply_multi_layer(self.critic_hidden, v_hid)

        a_hid = state
        if self.actor_hidden is not None:
            a_hid = apply_multi_layer(self.actor_hidden, a_hid)

        a = self.tanh(self.actor(a_hid))
        value = self.critic(v_hid).squeeze(-1)
        return a, value


def collect_trajectories(envs, policy, tmax=200, nrand=5):
    global i_episode
    global writer

    episode_rewards = 0

    def to_tensor(x, dtype=np.float32):
        return torch.from_numpy(np.array(x).astype(dtype)).to(device)

    # initialize returning lists and start the game!
    state_list = []
    reward_list = []
    prob_list = []
    action_list = []
    value_list = []
    done_list = []

    envs.reset()

    for _ in range(nrand):
        action = np.random.randn(num_agents, action_size)
        action = np.clip(action, -1.0, 1.0)
        _, reward, done, _ = envs.step(action)
        reward = torch.tensor(reward, device=device)

    for t in range(tmax):
        states = envs.get_screen().to(device)
        action_est, values = policy(states)
        sigma = nn.Parameter(torch.zeros(action_size))
        dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=-1).detach()
        values = values.detach()
        actions = actions.detach()

        env_actions = actions.cpu().numpy()
        _, reward, done, _ = envs.step(env_actions)
        rewards = to_tensor(reward)
        dones = to_tensor(done)

        state_list.append(states.unsqueeze(0))
        prob_list.append(log_probs.unsqueeze(0))
        action_list.append(actions.unsqueeze(0))
        reward_list.append(rewards.unsqueeze(0))
        value_list.append(values.unsqueeze(0))
        done_list.append(dones)

        if np.any(dones.cpu().numpy()):
            episode_rewards += rewards.sum(dim=0)
            i_episode += dones.sum(dim=0)
            writer.add_scalar(
                "Episodes average rewards",
                episode_rewards.item() / dones.sum(dim=0).item(),
                i_episode.item(),
            )
            state = envs.reset()
            episode_rewards = 0

    state_list = torch.cat(state_list, dim=0)
    prob_list = torch.cat(prob_list, dim=0)
    action_list = torch.cat(action_list, dim=0)
    reward_list = torch.cat(reward_list, dim=0)
    value_list = torch.cat(value_list, dim=0)
    done_list = torch.cat(done_list, dim=0)
    return prob_list, state_list, action_list, reward_list, value_list, done_list


def calc_returns(rewards, values, dones):
    n_step = len(rewards)
    n_agent = len(rewards[0])

    GAE = torch.zeros(n_step, n_agent).float().to(device)
    returns = torch.zeros(n_step, n_agent).float().to(device)

    GAE_current = torch.zeros(n_agent).float().to(device)

    TAU = 0.95
    discount = 0.99
    values_next = values[-1].detach()
    returns_current = values[-1].detach()
    for irow in reversed(range(n_step)):
        values_current = values[irow]
        rewards_current = rewards[irow]
        gamma = discount * (1.0 - dones[irow].float())

        td_error = rewards_current + gamma * values_next - values_current
        GAE_current = td_error + gamma * TAU * GAE_current
        returns_current = rewards_current + gamma * returns_current
        GAE[irow] = GAE_current
        returns[irow] = returns_current
        values_next = values_current

    return GAE, returns


def get_screen():
    screen = env._get_observation().transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)


def eval_policy(envs, policy, tmax=1000):
    reward_list = []
    state = envs.reset()
    for t in range(tmax):
        states = get_screen()
        action_est, values = policy(states)
        sigma = nn.Parameter(torch.zeros(action_size))
        dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))
        actions = dist.sample()
        _, reward, done, _ = envs.step(actions[0])
        dones = done
        reward_list.append(np.mean(reward))
        if np.any(dones):
            break
    return reward_list

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.linear = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear(x.view(x.size(0), -1)))
        return self.head(x)


def get_screen():
    global stacked_screens
    screen = env._get_observation().transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return preprocess(screen).unsqueeze(0).to("cpu")


def actionSelect(state, i_episode):
    global steps_done
    global eps_threshold
    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START - i_episode / EPS_DECAY_LAST_FRAME)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )


def modelOptmize():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__ == "__main__":
    
    if args.a2c:
        resize = T.Compose([T.ToPILImage(), T.Resize(40), T.ToTensor()])
    envs = make_batch_env(test=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[+] Using device: {}".format(device))

    envs.reset()

    num_agents = envs.num_envs
    print("Number of agents:", num_agents)

    init_screen = envs.get_screen().to(device)
    _, _, screen_height, screen_width = init_screen.shape

    action_size = envs.action_space.shape[0]
    print("Size of each action:", action_size)

    plt.figure()
    plt.imshow(
        init_screen[0].cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation="none"
    )
    plt.title("Example extracted screen")
    plt.show()

    writer = SummaryWriter()
    i_episode = 0

    policy = A2C(
        state_size=(screen_height, screen_width),
        action_size=action_size,
        shared_layers=[128, 64],
        critic_hidden_layers=[64],
        actor_hidden_layers=[64],
        init_type="xavier-uniform",
        seed=0,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=2e-4)

    PATH = "a2cpolicy.pt"

    best_mean_reward = None

    scores_window = deque(maxlen=100)  # last 100 scores

    discount = 0.9
    epsilon = 0.09
    beta = 0.01
    opt_epoch = 10
    season = 10000
    batch_size = 64
    tmax = 1000 // num_agents  # env episode steps
    save_scores = []
    start_time = timeit.default_timer()

    for s in tqdm.tqdm(range(season)):
        policy.eval()
        (
            old_probs_lst,
            states_lst,
            actions_lst,
            rewards_lst,
            values_lst,
            dones_list,
        ) = collect_trajectories(envs=envs, policy=policy, tmax=tmax, nrand=5)

        season_score = rewards_lst.sum(dim=0).sum().item()
        scores_window.append(season_score)
        save_scores.append(season_score)

        gea, target_value = calc_returns(
            rewards=rewards_lst, values=values_lst, dones=dones_list
        )
        gea = (gea - gea.mean()) / (gea.std() + 1e-8)

        policy.train()

        # cat all agents
        def concat_all(v):
            # print(v.shape)
            if len(v.shape) == 3:  # actions
                return v.reshape([-1, v.shape[-1]])
            if len(v.shape) == 5:  # states
                v = v.reshape([-1, v.shape[-3], v.shape[-2], v.shape[-1]])
                # print(v.shape)
                return v
            return v.reshape([-1])

        old_probs_lst = concat_all(old_probs_lst)
        states_lst = concat_all(states_lst)
        actions_lst = concat_all(actions_lst)
        rewards_lst = concat_all(rewards_lst)
        values_lst = concat_all(values_lst)
        gea = concat_all(gea)
        target_value = concat_all(target_value)

        # gradient ascent step
        n_sample = len(old_probs_lst) // batch_size
        idx = np.arange(len(old_probs_lst))
        np.random.shuffle(idx)
        for epoch in range(opt_epoch):
            for b in range(n_sample):
                ind = idx[b * batch_size : (b + 1) * batch_size]
                g = gea[ind]
                tv = target_value[ind]
                actions = actions_lst[ind]
                old_probs = old_probs_lst[ind]

                action_est, values = policy(states_lst[ind])
                sigma = nn.Parameter(torch.zeros(action_size))
                dist = torch.distributions.Normal(
                    action_est, F.softplus(sigma).to(device)
                )
                log_probs = dist.log_prob(actions)
                log_probs = torch.sum(log_probs, dim=-1)
                entropy = torch.sum(dist.entropy(), dim=-1)

                ratio = torch.exp(log_probs - old_probs)
                ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
                L_CLIP = torch.mean(torch.min(ratio * g, ratio_clipped * g))
                S = entropy.mean()
                L_VF = 0.5 * (tv - values).pow(2).mean()

                L = -(L_CLIP - L_VF + beta * S)
                optimizer.zero_grad()

                L.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()
                del L

        epsilon *= 0.99

        beta *= 0.99

        mean_reward = np.mean(scores_window)
        writer.add_scalar("epsilon", epsilon, s)
        writer.add_scalar("beta", beta, s)
        writer.add_scalar("Score", mean_reward, s)
        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epsilon": epsilon,
                    "beta": beta,
                },
                PATH,
            )
            if best_mean_reward is not None:
                print(
                    "Best mean reward updated %.3f -> %.3f, model saved"
                    % (best_mean_reward, mean_reward)
                )
            best_mean_reward = mean_reward
        if s >= 25 and mean_reward > 50:
            print(
                "Environment solved in {:d} seasons!\tAverage Score: {:.2f}".format(
                    s + 1, mean_reward
                )
            )
            break

    print("Average Score: {:.2f}".format(mean_reward))
    elapsed = timeit.default_timer() - start_time
    print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
    writer.close()
    envs.close()

    fig = plt.figure()
    plt.plot(np.arange(len(save_scores)), save_scores)
    plt.ylabel("Score")
    plt.xlabel("Season #")
    plt.grid()
    plt.show()

    if args.EVAL:
        print("Evaluating the model...")
        episode = 10
        scores_window = deque(maxlen=100)  # last 100 scores
        env = KukaDiverseObjectEnv(
            renders=False,
            isDiscrete=False,
            removeHeightHack=False,
            maxSteps=20,
            isTest=True,
        )
        env.cid = p.connect(p.DIRECT)
        checkpoint = torch.load(PATH)
        policy.load_state_dict(checkpoint["policy_state_dict"])

        for e in range(episode):
            rewards = eval_policy(envs=env, policy=policy)
            reward = np.sum(rewards, 0)
            print("Episode: {0:d}, reward: {1}".format(e + 1, reward), end="\n")

    if args.ppo:
        env = KukaDiverseObjectEnv(
        renders=False, isDiscrete=False, removeHeightHack=False, maxSteps=20
    )
    env.cid = p.connect(p.DIRECT)
    action_space = spaces.Box(low=-1, high=1, shape=(5, 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[+] Using device: {}".format(device))

    env.reset()

    num_agents = 1
    print("Number of agents:", num_agents)

    resize = T.Compose(
        [T.ToPILImage(), T.Resize(40, interpolation=Image.BICUBIC), T.ToTensor()]
    )
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape

    action_size = env.action_space.shape[0]
    print("Size of each action:", action_size)

    plt.figure()
    plt.imshow(
        init_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation="none"
    )
    plt.title("Example")
    plt.show()

    writer = SummaryWriter()
    i_episode = 0
    ten_rewards = 0

    policy = A2C(
        state_size=(screen_height, screen_width),
        action_size=action_size,
        shared_layers=[64, 32],
        critic_hidden_layers=[32],
        actor_hidden_layers=[32],
        seed=0,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=2e-4)

    PATH = "PPOPolicy.pt"

    writer = SummaryWriter()
    best_mean_reward = None

    scores_window = deque(maxlen=100)  # last 100 scores

    discount = 0.99
    epsilon = 0.09
    beta = 0.02
    opt_epoch = 50

    episodes = 1000000
    batch_size = 128
    tmax = 1000  # env episode steps
    save_scores = []
    start_time = timeit.default_timer()

    for e in tqdm.tqdm(range(episodes)):
        policy.eval()
        (
            old_probs_lst,
            states_lst,
            actions_lst,
            rewards_lst,
            values_lst,
            dones_list,
        ) = collect_trajectories(envs=env, policy=policy, tmax=tmax, nrand=5)

        episodes_score = rewards_lst.sum(dim=0).item()
        scores_window.append(episodes_score)
        save_scores.append(episodes_score)

        gea, target_value = calc_returns(
            rewards=rewards_lst, values=values_lst, dones=dones_list
        )
        gea = (gea - gea.mean()) / (gea.std() + 1e-8)

        policy.train()

        def concat_all(v):
            if len(v.shape) == 3:  # actions
                return v.reshape([-1, v.shape[-1]])
            if len(v.shape) == 5:  # states
                v = v.reshape([-1, v.shape[-3], v.shape[-2], v.shape[-1]])
                return v
            return v.reshape([-1])

        old_probs_lst = concat_all(old_probs_lst)
        states_lst = concat_all(states_lst)
        actions_lst = concat_all(actions_lst)
        rewards_lst = concat_all(rewards_lst)
        values_lst = concat_all(values_lst)
        gea = concat_all(gea)
        target_value = concat_all(target_value)

        n_sample = len(old_probs_lst) // batch_size
        idx = np.arange(len(old_probs_lst))
        np.random.shuffle(idx)
        for epoch in range(opt_epoch):
            for b in range(n_sample):
                ind = idx[b * batch_size : (b + 1) * batch_size]
                g = gea[ind]
                tv = target_value[ind]
                actions = actions_lst[ind]
                old_probs = old_probs_lst[ind]

                action_est, values = policy(states_lst[ind])
                sigma = nn.Parameter(torch.zeros(action_size))
                dist = torch.distributions.Normal(
                    action_est, F.softplus(sigma).to(device)
                )
                log_probs = dist.log_prob(actions)
                log_probs = torch.sum(log_probs, dim=-1)
                entropy = torch.sum(dist.entropy(), dim=-1)

                ratio = torch.exp(log_probs - old_probs)
                ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
                L_CLIP = torch.mean(torch.min(ratio * g, ratio_clipped * g))
                S = entropy.mean()
                L_VF = 0.5 * (tv - values).pow(2).mean()
                L = -(L_CLIP - L_VF + beta * S)
                optimizer.zero_grad()
                L.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()
                del L

        epsilon *= 0.999

        beta *= 0.998

        mean_reward = np.mean(scores_window)
        writer.add_scalar("epsilon", epsilon, e)
        writer.add_scalar("beta", beta, e)
        # display some progress every n iterations
        if best_mean_reward is None or best_mean_reward < mean_reward:
            # For saving the model and possibly resuming training
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epsilon": epsilon,
                    "beta": beta,
                },
                PATH,
            )
            if best_mean_reward is not None:
                print(
                    "Best mean reward updated %.3f -> %.3f, model saved"
                    % (best_mean_reward, mean_reward)
                )
            best_mean_reward = mean_reward
        print(
            "Environment solved in {:d} episodess!\tAverage Score: {:.2f}".format(
                e + 1, mean_reward
            )
        )

    print("Average Score: {:.2f}".format(mean_reward))
    elapsed = timeit.default_timer() - start_time
    print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
    writer.close()
    env.close()

    fig = plt.figure()
    plt.plot(np.arange(len(save_scores)), save_scores)
    plt.ylabel("Score")
    plt.xlabel("episodes #")
    plt.grid()
    plt.show()

    if args.EVAL == True:
        episode = 10
        scores_window = deque(maxlen=100)  # last 100 scores
        env = KukaDiverseObjectEnv(
            renders=False,
            isDiscrete=False,
            removeHeightHack=False,
            maxSteps=20,
            isTest=True,
        )
        env.cid = p.connect(p.DIRECT)
        checkpoint = torch.load(PATH)
        policy.load_state_dict(checkpoint["policy_state_dict"])

        for e in range(episode):
            rewards = eval_policy(envs=env, policy=policy)
            reward = np.sum(rewards, 0)
            print("Episode: {0:d}, reward: {1}".format(e + 1, reward), end="\n")

    if args.dqn:
        
            plt.ion()

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[+] Using device: {}".format(device))
    STACK_SIZE = 5
    env = KukaDiverseObjectEnv(
        renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20
    )
    env.cid = p.connect(p.DIRECT)

    preprocess = T.Compose(
        [T.ToPILImage(), T.Grayscale(num_output_channels=1), T.Resize(40), T.ToTensor()]
    )
    env.reset()
    plt.figure()
    plt.imshow(
        get_screen().cpu().squeeze(0)[-1].numpy(), cmap="Greys", interpolation="none"
    )
    plt.title("Example extracted screen")
    plt.show()

    BATCH_SIZE = 64
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 400
    EPS_DECAY_LAST_FRAME = 10**5
    TARGET_UPDATE = 100
    LEARNING_RATE = 1e-3

    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape

    n_actions = env.action_space.n

    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(10000)

    eps_threshold = 0

    PATH = "dqnPolicy.pt"

    num_episodes = 10000000
    writer = SummaryWriter()
    total_rewards = []
    avg_rewards = []
    ten_rewards = 0
    best_mean_reward = None
    start_time = timeit.default_timer()
    for i_episode in tqdm.tqdm(range(num_episodes)):
        env.reset()
        state = get_screen()
        stacked_states = collections.deque(STACK_SIZE * [state], maxlen=STACK_SIZE)
        for t in count():
            stacked_states_t = torch.cat(tuple(stacked_states), dim=1)

            action = actionSelect(stacked_states_t, i_episode)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            next_state = get_screen()
            if not done:
                next_stacked_states = stacked_states
                next_stacked_states.append(next_state)
                next_stacked_states_t = torch.cat(tuple(next_stacked_states), dim=1)
            else:
                next_stacked_states_t = None

            memory.push(stacked_states_t, action, next_stacked_states_t, reward)

            stacked_states = next_stacked_states

            modelOptmize()
            if done:
                reward = reward.cpu().numpy().item()
                ten_rewards += reward
                total_rewards.append(reward)
                mean_reward = np.mean(total_rewards[-100:]) * 100
                avg_rewards.append(mean_reward)
                writer.add_scalar("epsilon", eps_threshold, i_episode)
                if (
                    best_mean_reward is None or best_mean_reward < mean_reward
                ) and i_episode > 100:
                    torch.save(
                        {
                            "policy_net_state_dict": policy_net.state_dict(),
                            "target_net_state_dict": target_net.state_dict(),
                            "optimizer_policy_net_state_dict": optimizer.state_dict(),
                        },
                        PATH,
                    )
                    if best_mean_reward is not None:
                        print(
                            "Best mean reward updated %.1f -> %.1f, model saved"
                            % (best_mean_reward, mean_reward)
                        )
                    best_mean_reward = mean_reward
                break

        if i_episode % 10 == 0:
            writer.add_scalar(
                "ten episodes average rewards", ten_rewards / 10.0, i_episode
            )
            ten_rewards = 0
        print(
            "Environment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                i_episode + 1, mean_reward
            )
        )
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    plt.plot(total_rewards)
    plt.show()
    plt.plot(avg_rewards)
    plt.show()
    print("Average Score: {:.2f}".format(mean_reward))
    elapsed = timeit.default_timer() - start_time
    print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
    writer.close()
    env.close()

    if args.EVAL:
        episode = 10
        scores_window = collections.deque(maxlen=100)  # last 100 scores
        env = KukaDiverseObjectEnv(
            renders=False,
            isDiscrete=True,
            removeHeightHack=False,
            maxSteps=20,
            isTest=True,
        )
        env.cid = p.connect(p.DIRECT)
        # load the model
        checkpoint = torch.load(PATH)
        policy_net.load_state_dict(checkpoint["policy_net_state_dict"])

        # evaluate the model
        for i_episode in range(episode):
            env.reset()
            state = get_screen()
            stacked_states = collections.deque(STACK_SIZE * [state], maxlen=STACK_SIZE)
            for t in count():
                stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
                # Select and perform an action
                action = policy_net(stacked_states_t).max(1)[1].view(1, 1)
                _, reward, done, _ = env.step(action.item())
                # Observe new state
                next_state = get_screen()
                stacked_states.append(next_state)
                if done:
                    break
            print("Episode: {0:d}, reward: {1}".format(i_episode + 1, reward), end="\n")