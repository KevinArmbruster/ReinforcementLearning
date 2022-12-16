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
import itertools

# Load packages
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import Agent, RandomAgent, DQNAgent
from lab2.Helper import Experience


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


def simulate(N_episodes, agent: Agent, buffer, early_stopping=230):
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

        # early stopping, if agent performs well
        ravg = running_average(episode_reward_list, n_ep_running_average)[-1]
        if early_stopping and ravg > early_stopping:
            print("Early Stopping, Agent performs well")
            break

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


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def hyper_parameter_search(hs_config, random_agent, n_ep_running_average):
    search_space = list(product_dict(**hs_config))
    print("Searching possibilities: ", len(search_space))

    results = []
    results1 = []
    SEARCH = trange(len(search_space), desc='Episode: ', leave=True)
    for i in SEARCH:
        print(search_space[i])
        hs_dqn_agent = DQNAgent(**search_space[i])
        simulate(int(search_space[i]["buffer_size"] / 80 * rough_fill_percentage), random_agent, hs_dqn_agent.buffer)
        episode_reward_list, episode_number_of_steps = simulate(search_space[i]["N_episodes"], hs_dqn_agent,
                                                                hs_dqn_agent.buffer)
        avg_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
        avg_steps = running_average(episode_number_of_steps, n_ep_running_average)[-1]
        results.append(avg_reward)
        results1.append(avg_steps)
        SEARCH.set_description(f"SEARCH {i} - Avg Reward/Steps: {avg_reward}/{avg_steps}")

    idx = np.argmax(results)
    best_hp = search_space[idx]
    print(idx, np.max(results), best_hp)
    return idx, results, results1, search_space


# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v2')
env.reset()

### Parameters
n_ep_running_average = 50

agent_config = {
    "N_episodes": 1000,  # advised range [100, 1000]
    "discount_factor": 99 / 100,
    "lr": 0.00055,  # 1e-3,  # advised range [1e-3, 1e-4]
    "n_actions": env.action_space.n,
    "dim_states": len(env.observation_space.high),
    "buffer_size": 10000,  # advised range [5000, 30000]
    "batch_size": 32,  # advised range [4, 128]
    "hidden_layer_sizes": [64, 64]  # advised 1-2 layers with 8-128 neurons
}

### Initialization
random_agent = RandomAgent(agent_config["n_actions"])
dqn_agent = DQNAgent(**agent_config)
# fill experience buffer
rough_fill_percentage = .1
# episode_reward_list, episode_number_of_steps = simulate(int(agent_config["buffer_size"] / 80 * rough_fill_percentage), random_agent, dqn_agent.buffer)
# plot_rewards_and_steps(episode_reward_list, episode_number_of_steps, "Random Agent")
print("Buffer size = ", dqn_agent.buffer.__len__())

### Hyperparameter Search
hs_config = {
    "N_episodes": [1000],  # np.linspace(100, 500, 3),
    "discount_factor": [49 / 50, 99 / 100, 499 / 500],
    "n_actions": [env.action_space.n],
    "dim_states": [len(env.observation_space.high)],
    "buffer_size": [10000],  # np.linspace(5000, 30000, 3).astype(int),
    "batch_size": [16, 32, 64],  # np.linspace(16, 128, 3).astype(int),
    "lr": np.linspace(1e-3, 1e-4, 3),
    "hidden_layer_sizes": [[64, 64], [128], ],
}
# idx, results, results1, search_space = hyper_parameter_search(hs_config, random_agent, n_ep_running_average)

### Training process
# print(agent_config)
# episode_reward_list, episode_number_of_steps = simulate(agent_config["N_episodes"], dqn_agent, dqn_agent.buffer)
# plot_rewards_and_steps(episode_reward_list, episode_number_of_steps, "DQN Agent")

### Save DQN
# torch.save(dqn_agent.main_q_network, "neural-network-1.pth")

### e)
# agent_config1 = {
#     "N_episodes": 300,  # advised range [100, 1000]
#     "discount_factor": 99 / 100,
#     "lr": 0.00055,  # 1e-3,  # advised range [1e-3, 1e-4]
#     "n_actions": env.action_space.n,
#     "dim_states": len(env.observation_space.high),
#     "buffer_size": 10000,  # advised range [5000, 30000]
#     "batch_size": 32,  # advised range [4, 128]
#     "hidden_layer_sizes": [64, 64]  # advised 1-2 layers with 8-128 neurons
# }
# dqn_agent1 = DQNAgent(**agent_config1)
# # fill experience buffer
# simulate(int(agent_config1["buffer_size"] / 80 * rough_fill_percentage), random_agent, dqn_agent1.buffer)
# episode_reward_list, episode_number_of_steps = simulate(agent_config1["N_episodes"], dqn_agent1, dqn_agent1.buffer, early_stopping=None)
# plot_rewards_and_steps(episode_reward_list, episode_number_of_steps, "DQN Agent - episodes = 300")
#
# agent_config2 = {
#     "N_episodes": 1500,  # advised range [100, 1000]
#     "discount_factor": 99 / 100,
#     "lr": 0.00055,  # 1e-3,  # advised range [1e-3, 1e-4]
#     "n_actions": env.action_space.n,
#     "dim_states": len(env.observation_space.high),
#     "buffer_size": 10000,  # advised range [5000, 30000]
#     "batch_size": 32,  # advised range [4, 128]
#     "hidden_layer_sizes": [64, 64]  # advised 1-2 layers with 8-128 neurons
# }
# dqn_agent2 = DQNAgent(**agent_config2)
# # fill experience buffer
# simulate(int(agent_config2["buffer_size"] / 80 * rough_fill_percentage), random_agent, dqn_agent2.buffer)
# episode_reward_list, episode_number_of_steps = simulate(agent_config2["N_episodes"], dqn_agent2, dqn_agent2.buffer, early_stopping=None)
# plot_rewards_and_steps(episode_reward_list, episode_number_of_steps, "DQN Agent - episodes = 1500")
#
#
# agent_config3 = {
#     "N_episodes": 1000,  # advised range [100, 1000]
#     "discount_factor": 99 / 100,
#     "lr": 0.00055,  # 1e-3,  # advised range [1e-3, 1e-4]
#     "n_actions": env.action_space.n,
#     "dim_states": len(env.observation_space.high),
#     "buffer_size": 5000,  # advised range [5000, 30000]
#     "batch_size": 32,  # advised range [4, 128]
#     "hidden_layer_sizes": [64, 64]  # advised 1-2 layers with 8-128 neurons
# }
# dqn_agent3 = DQNAgent(**agent_config3)
# # fill experience buffer
# simulate(int(agent_config3["buffer_size"] / 80 * rough_fill_percentage), random_agent, dqn_agent3.buffer)
# episode_reward_list, episode_number_of_steps = simulate(agent_config3["N_episodes"], dqn_agent3, dqn_agent3.buffer, early_stopping=230)
# plot_rewards_and_steps(episode_reward_list, episode_number_of_steps, "DQN Agent - buffer_size = 5000")
#
# agent_config4 = {
#     "N_episodes": 1000,  # advised range [100, 1000]
#     "discount_factor": 99 / 100,
#     "lr": 0.00055,  # 1e-3,  # advised range [1e-3, 1e-4]
#     "n_actions": env.action_space.n,
#     "dim_states": len(env.observation_space.high),
#     "buffer_size": 30000,  # advised range [5000, 30000]
#     "batch_size": 32,  # advised range [4, 128]
#     "hidden_layer_sizes": [64, 64]  # advised 1-2 layers with 8-128 neurons
# }
# dqn_agent4 = DQNAgent(**agent_config4)
# # fill experience buffer
# simulate(int(agent_config4["buffer_size"] / 80 * rough_fill_percentage), random_agent, dqn_agent4.buffer)
# episode_reward_list, episode_number_of_steps = simulate(agent_config4["N_episodes"], dqn_agent4, dqn_agent4.buffer, early_stopping=230)
# plot_rewards_and_steps(episode_reward_list, episode_number_of_steps, "DQN Agent - buffer_size = 30000")

### g)
# model = torch.load('neural-network-1.pth')
# heights = np.arange(0, 1.5, 0.1)
# angles = np.arange(-np.pi, np.pi, 0.1)
# prod = np.array(list(itertools.product(heights, angles)))
# zeros = np.zeros(len(prod))
# heights = prod[:, 0]
# angles = prod[:, 1]
#
# states = np.asarray([zeros, heights, zeros, zeros, angles, zeros, zeros, zeros], dtype=np.float32).T
# q = model(torch.from_numpy(states))
# value, actions = torch.max(q, axis=1)
#
#
# def d3_plot(x, y, z, xlabel, ylabel, zlabel, title, lim=False):
#     ax = plt.figure().add_subplot(projection='3d')
#
#     colors = ('c', 'm', 'b', 'g')
#     c_list = []
#     for a in actions:
#         c_list.extend(colors[a])
#
#     ax.scatter(xs=x, ys=y, zs=z, zdir='z', c=c_list)
#
#     # ax.legend()
#     ax.zaxis.set_major_locator(MaxNLocator(integer=True))
#     if lim:
#         ax.set_zlim(0, 3)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_zlabel(zlabel)
#
#     ax.set_title(title)
#
#     ax.view_init(elev=10., azim=-15, roll=0)
#     plt.show()
#
#
# d3_plot(heights, angles, actions, 'Height', 'Angle', 'Action', "Policy in restricted state space", True)
# d3_plot(heights, angles, value.detach(), 'Height', 'Angle', 'Value', "Value function in restricted state space")
