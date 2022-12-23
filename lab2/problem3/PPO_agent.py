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
# Last update: 29th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn
from collections import deque

class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''
    def __init__(self, nn_actor):
        self.actor = torch.load(nn_actor)

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        mu, sigma = self.actor(torch.tensor(state))     # The tensors of mean and variance
        mu = mu.detach().numpy()
        std = np.sqrt(sigma.detach().numpy())
        actions = np.clip([np.random.normal(mu[0], std[0]), 
                          np.random.normal(mu[1], std[1])],
                          -1, 1)
        return actions

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent():
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int, nn_actor):
        self.n_actions = n_actions

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)


class ExperienceReplayBuffer(object):

    def __init__(self, maximum_length):
        self.buffer = deque(maxlen=int(maximum_length))

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n):
        #n = int(n)
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        indices = np.random.choice(len(self.buffer), size=n, replace=False)
        batch = [self.buffer[i] for i in indices]

        # convert a list of tuples into a tuple of list we do zip(*batch)
        return zip(*batch)

    def unzip_buffer(self):
        return zip(*self.buffer)


class Critic_network(nn.Module):
  def __init__(self, dim_state, dev):
    super().__init__()

    # Number of neurons
    neu_1 = 400
    neu_2 = 200

    # Three layers
    self.layer_1 = nn.Linear(dim_state, neu_1, device = dev)  # Input layer
    self.layer_2 = nn.Linear(neu_1, neu_2, device = dev)                    # Middle layer  
    self.layer_3 = nn.Linear(neu_2, 1, device = dev)                        # Output layer  

  def forward(self, s):
    output_1 = torch.relu(self.layer_1(s))                       # Output of the input layer
    output_2 = torch.relu(self.layer_2(output_1))                # Output of the middle layer
    output_3 = self.layer_3(output_2)                            # Output of the last layer
    
    return output_3


class Actor_network(nn.Module):
  def __init__(self, dim_state, dim_action, dev):
    super().__init__()

    # Number of neurons
    neu_1 = 400
    neu_2 = 200

    # Input layer
    self.input_layer = nn.Linear(dim_state, neu_1, device = dev)

    # Middle layer of mean and variance 
    self.mu_hidden = nn.Linear(neu_1, neu_2, device = dev)
    self.sigma_hidden = nn.Linear(neu_1, neu_2, device = dev)

    # Output layer of mean and variance
    self.mu_output = nn.Linear(neu_2, dim_action, device = dev)
    self.sigma_output = nn.Linear(neu_2, dim_action, device = dev)

  def forward(self, s):
    # Output of the input layer
    output_1 = torch.relu(self.input_layer(s))

    # Output of the hidden layer of mean and variance
    output_mu_hidden = torch.relu(self.mu_hidden(output_1))
    output_sigma_hidden = torch.relu(self.sigma_hidden(output_1))

    # Output of the last layer of mean and variance
    output_mu = torch.tanh(self.mu_output(output_mu_hidden))
    output_sigma = torch.sigmoid(self.sigma_output(output_sigma_hidden))

    return output_mu, output_sigma


class PPOAgent(object):
  def __init__(self, gamma, epsilon, alpha_critic, alpha_actor, dim_action, dim_state, dev):
    self.gamma = gamma                # Discount factor
    self.epsilon = epsilon
    self.alpha_critic = alpha_critic  # Learning rate for critic network
    self.alpha_actor = alpha_actor    # Learning rate for actor network
    self.dim_action = dim_action
    self.dim_state = dim_state
    self.dev = dev

    # Critic and actor network
    self.critic_nw = Critic_network(self.dim_state, self.dev)
    self.actor_nw = Actor_network(self.dim_state, self.dim_action, self.dev)

    # Adam optimizing
    self.optim_critic = torch.optim.Adam(self.critic_nw.parameters(), lr=self.alpha_critic)
    self.optim_actor = torch.optim.Adam(self.actor_nw.parameters(), lr=self.alpha_actor)

  def forward(self, s):
    mu, sigma = self.actor_nw(torch.tensor(s))   # The tensors of mean and variance
    mu = mu.detach().numpy()
    sigma = sigma.detach().numpy()    
    std = np.sqrt(sigma)
    actions = np.clip([np.random.normal(mu[0], std[0]), 
                      np.random.normal(mu[1], std[1])], -1, 1)
    return actions

  def gaussian_pdf(self, actions, mu, sigma):
    pi_1 = (1/torch.sqrt(2*np.pi*sigma[:,0])) * torch.exp(-(actions[:,0]-mu[:,0])**2 / (2*sigma[:,0]))
    pi_2 = (1/torch.sqrt(2*np.pi*sigma[:,1])) * torch.exp(-(actions[:,1]-mu[:,1])**2 / (2*sigma[:,1]))
    pi = pi_1*pi_2
    return pi

  def update(self, M, buffer):
    state, action, reward, next_state, done = buffer.unzip_buffer() 

    # Calculate target value
    G_i = np.zeros(len(state))
    G_i[-1] = reward[-1]
    for t in reversed(range(len(state)-1)):
      G_i[t]= reward[t] + G_i[t+1]*self.gamma
    G_i = torch.tensor(G_i, dtype = torch.float32) 

    gradient_state = torch.tensor(state, requires_grad = True)
    gradient_action = torch.tensor(action, requires_grad = True)

    mu_old, sigma_old = self.actor_nw(gradient_state)
    pi_old = self.gaussian_pdf(gradient_action, mu_old, sigma_old).detach()

    for n in range(M):
      self.optim_critic.zero_grad()
      state_values = self.critic_nw(gradient_state).squeeze()
      loss = nn.functional.mse_loss(state_values, G_i)
      loss.backward()
      nn.utils.clip_grad_norm_(self.critic_nw.parameters(), max_norm=1)
      self.optim_critic.step()

      self.optim_actor.zero_grad()
      state_tensor = torch.tensor(state)
      V_w = self.critic_nw(state_tensor).squeeze()
      Psi = G_i - V_w
      mu_new, sigma_new = self.actor_nw(gradient_state)
      pi_new = self.gaussian_pdf(gradient_action, mu_new, sigma_new)
      r = pi_new / pi_old


      product_temp = torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon) * Psi
      min_temp = torch.min(r*Psi, product_temp)
      loss = -torch.mean(min_temp)
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor_nw.parameters(),max_norm=1)
      self.optim_actor.step()