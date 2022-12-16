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
# Last update: 26th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch

from lab2.Helper import ActorPolicyNetwork, CriticValueNetwork, ExperienceReplayBuffer
from lab2.problem2.DDPG_soft_updates import soft_updates


class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def forward(self, state: torch.Tensor):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: torch.Tensor) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)


class DDPGAgent(Agent):
    def __init__(self, dim_actions: int, dim_states: int, hidden_layer_sizes: list, lr_actor: float, lr_critic: float,
                 batch_size: int, buffer_size: int, noise_mu: float, noise_sigma: float, discount_factor: float,
                 update_freq: int, update_const: float, **kwargs):
        super(DDPGAgent, self).__init__(dim_actions)

        self.dim_actions = dim_actions
        self.dim_states = dim_states
        self.discount_factor = discount_factor
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.update_freq = update_freq
        self.update_const = update_const

        ### Noise
        self.time_step = 0
        self.previous_exploration_noise = None
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma ** 2 * np.identity(dim_actions)

        ### Buffer
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = ExperienceReplayBuffer(maximum_length=buffer_size)

        ### Networks
        self.main_actor_network = ActorPolicyNetwork(dim_states=dim_states, dim_actions=dim_actions,
                                                     hidden_layer_sizes=hidden_layer_sizes, lr=lr_actor)
        self.target_actor_network = ActorPolicyNetwork(dim_states=dim_states, dim_actions=dim_actions,
                                                       hidden_layer_sizes=hidden_layer_sizes, lr=lr_actor)

        self.main_critic_network = CriticValueNetwork(dim_states=dim_states, dim_actions=dim_actions,
                                                      hidden_layer_sizes=hidden_layer_sizes, lr=lr_critic)
        self.target_critic_network = CriticValueNetwork(dim_states=dim_states, dim_actions=dim_actions,
                                                        hidden_layer_sizes=hidden_layer_sizes, lr=lr_critic)

        soft_updates(network=self.main_actor_network, target_network=self.target_actor_network, tau=1)
        soft_updates(network=self.main_critic_network, target_network=self.target_critic_network, tau=1)

    def forward(self, state: torch.Tensor) -> np.ndarray:
        # predict next action + exploration_noise
        exploration_noise = self.ornstein_uhlenbeck_noise()
        action = self.main_actor_network(state)
        return torch.clip(action + exploration_noise, -1, 1).detach().numpy()

    def ornstein_uhlenbeck_noise(self):
        if self.previous_exploration_noise is None:
            self.previous_exploration_noise = 0
            return self.previous_exploration_noise
        else:
            gauss = np.random.multivariate_normal(mean=np.zeros(self.dim_actions), cov=self.noise_sigma)
            self.previous_exploration_noise = -self.noise_mu * self.previous_exploration_noise + gauss
            return torch.from_numpy(self.previous_exploration_noise)

    def backward(self):
        batch = self.buffer.sample_batch(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        states = torch.from_numpy(np.array(states, dtype=np.float32))
        actions = torch.from_numpy(np.array(actions, dtype=np.float32))
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
        dones = torch.tensor(np.invert(dones), dtype=torch.float32)  # invert dones
        rewards = torch.tensor(rewards, dtype=torch.float32)

        ### Update Critic Value Network
        # Targets for Critic
        target_action = self.target_actor_network(next_states)
        target_Q_value = torch.squeeze(self.target_critic_network(next_states, target_action.detach()))
        Q_targets = rewards + self.discount_factor * dones * target_Q_value
        assert (Q_targets.shape == (self.batch_size,))

        # current Q value of selected actions
        Q_value = torch.squeeze(self.main_critic_network(states, actions))
        assert (Q_value.shape == (self.batch_size,))

        # update critic
        self.main_critic_network.backward(Q_value, Q_targets)

        if self.time_step % self.update_freq == 0:
            ### Update Actor Policy Network
            policy_loss = -self.main_critic_network(states, self.main_actor_network(states)).mean()
            self.main_actor_network.backward(policy_loss)

            ### Update Target Networks
            soft_updates(network=self.main_actor_network, target_network=self.target_actor_network,
                         tau=self.update_const)
            soft_updates(network=self.main_critic_network, target_network=self.target_critic_network,
                         tau=self.update_const)
        self.time_step += 1
