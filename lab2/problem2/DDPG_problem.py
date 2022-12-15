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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

import gym
import matplotlib.pyplot as plt
# Load packages
import numpy as np
from tqdm import trange

from DDPG_agent import Agent, RandomAgent
from lab2.problem1.Helper import ExperienceReplayBuffer, Experience


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


def simulate(agent: Agent, buffer: ExperienceReplayBuffer = None, N_episodes=200):
    episode_reward_list = []  # this list contains the total reward per episode
    episode_number_of_steps = []  # this list contains the number of steps per episode

    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
    for i in EPISODES:
        # Reset enviroment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0

        while not done:
            # Take a random action
            action = agent.forward(state)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)

            if buffer:
                exp = Experience(state, action, reward, next_state, done)
                buffer.append(exp)

                agent.backward()

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1

        # Append episode reward
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


def plot_rewards_and_steps(episode_reward_list, episode_number_of_steps, title=""):
    episodes = len(episode_reward_list)
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, episodes + 1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, episodes + 1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot([i for i in range(1, episodes + 1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, episodes + 1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.suptitle(title)
    plt.show()


# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
n_ep_running_average = 50  # Running average of 50 episodes
agent_config = {
    "N_episodes": 1000,  # advised range [100, 1000]
    "discount_factor": 0.95,
    "lr_actor": 0.00055,  # 1e-3,  # advised range [1e-3, 1e-4]
    "lr_critic": 0.00055,  # 1e-3,  # advised range [1e-3, 1e-4]
    "n_actions": len(env.action_space.high),
    "n_states": len(env.observation_space.high),
    "buffer_size": 10000,  # advised range [5000, 30000]
    "batch_size": 32,  # advised range [4, 128]
    "hidden_layer_sizes": [64, 64]  # advised 1-2 layers with 8-128 neurons
}

# Agent initialization
rnd_agent = RandomAgent(agent_config["n_actions"])

# Training process
episode_reward_list, episode_number_of_steps = simulate(rnd_agent, None, 100)

plot_rewards_and_steps(episode_reward_list, episode_number_of_steps)
