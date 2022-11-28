# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import *
from lab2.problem1.Helper import Experience, ExperienceReplayBuffer, DQNNetwork


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def simulate(N_episodes, agent, buffer):
    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []  # this list contains the total reward per episode
    episode_number_of_steps = []  # this list contains the number of steps per episode

    # trange is an alternative to range in python, from the tqdm library
    # It shows a nice progression bar that you can update with useful information
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0

        while not done:
            # Take a random action
            action = agent.forward(state=state, N=N_episodes, k=i)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, *_ = env.step(action)

            exp = Experience(state, action, reward, next_state, done)
            buffer.append(exp)

            agent.backward()

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                i, total_episode_reward, t,
                running_average(episode_reward_list, n_ep_running_average)[-1],
                running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    return episode_reward_list, episode_number_of_steps


def plot_rewards_and_steps(N_episodes, episode_reward_list, episode_number_of_steps):
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, N_episodes + 1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, N_episodes + 1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot([i for i in range(1, N_episodes + 1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, N_episodes + 1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()


# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v2')
env.reset()

### Parameters
N_episodes = 100  # advised range [100, 1000]
discount_factor = 49 / 50
n_ep_running_average = 50
n_actions = env.action_space.n  # Number of available actions
n_states = len(env.observation_space.high)  # State dimensionality
buffer_size = 5000  # advised range [5000, 30000]
batch_size = 4  # advised range [4, 128]

### Initialization
random_agent = RandomAgent(n_actions)
dqn_agent = DQNAgent(n_actions=n_actions, n_states=n_states, buffer_size=buffer_size, discount_factor=discount_factor,
                     batch_size=batch_size)
# fill experience buffer
simulate(int(buffer_size / 50), random_agent, dqn_agent.buffer)
print("Buffer size = ", dqn_agent.buffer.__len__())

### Training process
episode_reward_list, episode_number_of_steps = simulate(N_episodes, dqn_agent, dqn_agent.buffer)

plot_rewards_and_steps(N_episodes, episode_reward_list, episode_number_of_steps)
