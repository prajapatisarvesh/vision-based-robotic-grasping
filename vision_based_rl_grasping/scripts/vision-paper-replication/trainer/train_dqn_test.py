import numpy as np
from models.dqn import DQN
import gym
import torch
from models.replay_memory import ReplayMemory
from trainer.train_dqn import train_dqn_batch
import copy
import tqdm
import gym
from models.exponential_scheduling import ExponentialSchedule
import os
import matplotlib.pyplot as plt
import gym
from gym.envs import register as register_env
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_dqn(
    env,
    num_steps,
    *,
    num_saves=5,
    replay_size,
    replay_prepopulate_steps=0,
    batch_size,
    exploration,
    gamma,
):
    """
    DQN algorithm.

    Compared to previous training procedures, we will train for a given number
    of time-steps rather than a given number of episodes.  The number of
    time-steps will be in the range of millions, which still results in many
    episodes being executed.

    Args:
        - env: The openai Gym environment
        - num_steps: Total number of steps to be used for training
        - num_saves: How many models to save to analyze the training progress.
        - replay_size: Maximum size of the ReplayMemory
        - replay_prepopulate_steps: Number of steps with which to prepopulate
                                    the memory
        - batch_size: Number of experiences in a batch
        - exploration: a ExponentialSchedule

        - gamma: The discount factor

    Returns: (saved_models, returns)
        - saved_models: Dictionary whose values are trained DQN models
        - returns: Numpy array containing the return of each training episode
        - lengths: Numpy array containing the length of each training episode
        - losses: Numpy array containing the loss of each training batch
    """
    # check that environment states are compatible with our DQN representation
    
    assert (
        isinstance(env.observation_space, gym.spaces.Box)
        and len(env.observation_space.shape) == 1
    )

    # get the state_size from the environment
    state_size = env.observation_space.shape[0]

    # initialize the DQN and DQN-target models
    dqn_model = DQN(state_size, env.action_space.n).to(device)
    dqn_target = DQN.custom_load(dqn_model.custom_dump()).to(device)

    # initialize the optimizer
    optimizer = torch.optim.Adam(dqn_model.parameters())

    # initialize the replay memory and prepopulate it
    memory = ReplayMemory(replay_size, state_size)
    memory.populate(env, replay_prepopulate_steps)

    # initiate lists to store returns, lengths and losses
    rewards = []
    returns = []
    lengths = []
    losses = []

    # initiate structures to store the models at different stages of training
    t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)
    saved_models = {}

    i_episode = 0  # use this to indicate the index of the current episode
    t_episode = 0  # use this to indicate the time-step inside current episode

    state = env.reset()  # initialize state of first episode

    # iterate for a total of `num_steps` steps
    pbar = tqdm.trange(num_steps)

    for t_total in pbar:
        # use t_total to indicate the time-step from the beginning of training

        # save model
        if t_total in t_saves:
            model_name = f'{100 * t_total / num_steps:04.1f}'.replace('.', '_')
            saved_models[model_name] = copy.deepcopy(dqn_model)

        # YOUR CODE HERE:
        #  * sample an action from the DQN using epsilon-greedy
        #  * use the action to advance the environment by one step
        #  * store the transition into the replay memory
        
        eps = exploration.value(t_total)
        if np.random.rand() > eps:
            action = dqn_model(torch.tensor(state).to(device))
            action = torch.argmax(action).item()
        else:
            action = env.action_space.sample()
        # action = int(action)
        try:
            next_state, reward, done, _ = env.step(action)
        except Exception as e:
            next_state, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        memory.add(state, action, reward, next_state, done)

        # YOUR CODE HERE:  once every 4 steps,
        #  * sample a batch from the replay memory
        #  * perform a batch update (use the train_dqn_batch() method!)

        if t_total % 4 == 0:
            batch = memory.sample(batch_size)
            loss = train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma)
            losses.append(loss)

        # YOUR CODE HERE:  once every 10_000 steps,
        #  * update the target network (use the dqn_model.state_dict() and
        #    dqn_target.load_state_dict() methods!)

        if(t_total % 10000) == 0:
            dqn_target.load_state_dict(dqn_model.state_dict())

        if done:
            # YOUR CODE HERE:  anything you need to do at the end of an
            # episode, e.g. compute return G, store stuff, reset variables,
            # indices, lists, etc.

            state = env.reset()
            lengths.append(t_episode)
            G = 0
            for r in rewards[::-1]:
                G = r + gamma * G
            returns.append(G)
            
            rewards = []

            pbar.set_description(
                f'Episode: {i_episode} | Steps: {t_episode + 1} | Return: {G:5.2f} | Epsilon: {eps:4.2f}'
            )
            # print(f'\rEpisode: {i_episode} | Steps: {t_episode + 1} | Return: {G:5.2f} | Epsilon: {eps:4.2f}', end="")

            i_episode += 1
            t_episode = 0
        else:
            # YOUR CODE HERE:  anything you need to do within an episode
            t_episode += 1
            state = next_state

    saved_models['100_0'] = copy.deepcopy(dqn_model)
    # print(t_total)
    return (
        saved_models,
        np.array(returns),
        np.array(lengths),
        np.array(losses),
    )


if __name__ == '__main__':
    gamma = 0.99
    register_env(
        id="Kuka-v0",
        entry_point="pybullet_envs.bullet.kuka_diverse_object_gym_env:KukaDiverseObjectEnv",
    )
    env = gym.make(id='Kuka-v0', renders=True, isDiscrete=False)
    # we train for many time-steps;  as usual, you can decrease this during development / debugging.
    # but make sure to restore it to 1_500_000 before submitting.
    num_steps = 1_500_000
    num_saves = 5  # save models at 0%, 25%, 50%, 75% and 100% of training

    replay_size = 200_000
    replay_prepopulate_steps = 50_000

    batch_size = 64
    exploration = ExponentialSchedule(1.0, 0.01, 1_000_000)

    # this should take about 90-120 minutes on a generic 4-core laptop
    dqn_models_cartpole, returns_cartpole, lengths_cartpole, losses_cartpole = train_dqn(
        env,
        num_steps,
        num_saves=num_saves,
        replay_size=replay_size,
        replay_prepopulate_steps=replay_prepopulate_steps,
        batch_size=batch_size,
        exploration=exploration,
        gamma=gamma,
    )