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
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import random
import numpy as np
from copy import deepcopy
from lab2.problem1.Helper import *


class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray, N, k):
        pass

    def backward(self):
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray, N=None, k=None) -> int:
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


class DQNAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, n_actions: int, n_states: int, buffer_size: int, discount_factor: float,
                 batch_size: int = 4):
        super(DQNAgent, self).__init__(n_actions)
        self.n_states = n_states

        # MDP
        self.discount_factor = discount_factor
        self.exploration_rate_min = 0.05
        self.exploration_rate_max = 0.99
        self.exploration_rate_percentage = 0.9

        ## NN
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = ExperienceReplayBuffer(maximum_length=buffer_size)
        self.current_iteration = 0
        self.target_nn_update_frequency = int(buffer_size / batch_size)
        self.hidden_layer_sizes = [8, 8]  # advised 1-2 layers with 8-128 neurons
        self.main_dqn_network = DQNNetwork(n_states=n_states, n_actions=n_actions,
                                           hidden_layer_sizes=self.hidden_layer_sizes)
        self.target_dqn_network = DQNNetwork(n_states=n_states, n_actions=n_actions,
                                             hidden_layer_sizes=self.hidden_layer_sizes)  # ide errors if not done initially
        self.update_target_network()

    def exploration_rate(self, N, k):
        # N = Episodes, k = current episode
        Z = round(N * self.exploration_rate_percentage)
        tmp = self.exploration_rate_max * (self.exploration_rate_min / self.exploration_rate_max) ** ((k - 1) / (Z - 1))
        return max(self.exploration_rate_min, tmp)

    def update_target_network(self):
        # alternative might be to copy only weights??
        # https://androidkt.com/copy-pytorch-model-using-deepcopy-and-state_dict/
        self.target_dqn_network = deepcopy(self.main_dqn_network)
        self.target_dqn_network.setup_optimizer()  # else old reference is in place

    def forward(self, state: np.ndarray, N, k):
        self.last_action = self.e_greedy_policy(state, N, k)
        return self.last_action

    def e_greedy_policy(self, state, N, k):
        if np.random.rand() <= self.exploration_rate(N, k):
            action = random.randrange(self.n_actions)  # choose iid action
        else:
            action = self.main_dqn_network.forward(state)
            action = torch.argmax(action).item()
        return action

    def backward(self):
        try:
            batch = self.buffer.sample_batch(self.batch_size)
            states, actions, rewards, next_states, dones = batch
            states = np.array(states)

            # calc targets
            Q_target = self.target_dqn_network.forward(states)
            targets = rewards + self.discount_factor * dones[:, None] * Q_target

            # update main nn
            targets = np.array(targets)
            self.main_dqn_network.backward(targets, states)

            # update target nn
            self.current_iteration += 1
            if self.current_iteration >= self.target_nn_update_frequency:
                self.update_target_network()

        except IndexError as e:
            print("Could not train yet: ", e)
