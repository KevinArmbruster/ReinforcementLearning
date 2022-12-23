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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import itertools
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange
from PPO_agent import *
import matplotlib
from matplotlib.ticker import MaxNLocator
from google.colab import files


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

#dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev = torch.device("cpu")
# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
N_episodes = 1600                # Number of episodes to run for training
gamma = 0.99         # Discount factor
n_ep_running_average = 50      # Running average of 20 episodes
m = len(env.action_space.high) # dimensionality of the action
dim_s = len(env.observation_space.high) # dimensionality of the state
alpha_critic = 1e-3   # Learning rate of critic network
alpha_actor = 1e-5    # Learning_rate of actor network
L = 30000     # Buffer size
M = 10        # Epochs
epsilon = 0.2
actor_filename = 'neural-network-3-actor.pth'
critic_filename = 'neural-network-3-critic.pth'

# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

# Agent initialization
agent = PPOAgent(gamma, epsilon, alpha_critic, alpha_actor, m, dim_s, dev)

for i in EPISODES:
  # Reset enviroment data
  done = False
  state = env.reset()
  total_episode_reward = 0.
  t = 0

  buffer = ExperienceReplayBuffer(maximum_length=L)

  while not done:
    # Take a random action
    action = agent.forward(state)

    # Get next state and reward.  The done variable
    # will be True if you reached the goal position,
    # False otherwise
    next_state, reward, done, _ = env.step(action)
    buffer.append((state, action, reward, next_state, done))

    # Update episode reward
    total_episode_reward += reward

    # Update state for next iteration
    state = next_state
    t+= 1

  # Append episode reward
  episode_reward_list.append(total_episode_reward)
  episode_number_of_steps.append(t)

  # Close environment
  env.close()

  agent.update(M, buffer)
  
  EPISODES.set_description(
      "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
      i, total_episode_reward, t,
      running_average(episode_reward_list, n_ep_running_average)[-1],
      running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    
# Save PPO
torch.save(agent.actor_nw, "neural-network-3-actor.pth")
torch.save(agent.critic_nw, "neural-network-3-critic.pth")

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
#plt.show()
plt.savefig("figure.png")
files.download("figure.png")

'''# 3D plot
actor_network = torch.load('neural-network-3-actor.pth')
critic_network = torch.load('neural-network-3-critic.pth')

heights = np.linspace(0, 1.5, 100)
angles = np.linspace(-np.pi, np.pi, 100)
V = np.zeros((len(heights), len(angles)))
mu = np.zeros((len(heights), len(angles)))
Ys, Ws = np.meshgrid(heights, angles)

for y_idx, y in enumerate(heights):
    for w_idx, w in enumerate(angles):
        state = torch.tensor((0, y, 0, 0, w, 0, 0, 0), dtype=torch.float32)
        a = actor_network(state)
        mu[w_idx, y_idx] = a[0][1].item()
        V[w_idx, y_idx] = critic_network(torch.reshape(state, (1,-1))).item()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(Ws, Ys, V, cmap=mpl.cm.coolwarm)
ax.set_ylabel('height (y)')
ax.set_xlabel('angle (ω)')
ax.set_zlabel('V(s(y,ω))')
plt.show()

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(Ws, Ys, mu)
ax2.set_ylabel('height (y)')
ax2.set_xlabel('angle (ω)')
ax2.set_zlabel('μ(s,ω)')
plt.show()'''
