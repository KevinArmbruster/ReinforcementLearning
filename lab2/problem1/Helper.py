# Load packages
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn


class DQNNetwork(nn.Module):

    def __init__(self, n_states: int, n_actions: int, hidden_layer_sizes: list, lr=1e-3):
        super(DQNNetwork, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = None
        self.activation_functions = None
        self.optimizer = None
        self.lr = lr  # advised range [1e-3, 1e-4]
        self.max_grad_norm = 1.  # advised range [0.5, 2]

        self.setup_NN(n_states, hidden_layer_sizes, n_actions)
        self.setup_optimizer()

    def setup_NN(self, n_states, hidden_layer_sizes, n_actions, intermediate_af=nn.ReLU(), last_af=None):
        layers = nn.ModuleList()
        activation_functions = nn.ModuleList()

        layer_definitions = [n_states] + hidden_layer_sizes
        # layer_definitions.append(n_actions)

        for i in range(len(layer_definitions) - 1):
            layer = nn.Linear(layer_definitions[i], layer_definitions[i + 1])
            layers.append(layer)
            activation_functions.append(intermediate_af)

        # last layer with special
        layers.append(nn.Linear(layer_definitions[-1], n_actions))
        activation_functions.append(last_af)

        self.layers = layers
        self.activation_functions = activation_functions

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, state: np.ndarray):
        # state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)
        state_tensor = torch.from_numpy(state)
        # state_tensor = torch.tensor(state_tensor, requires_grad=True)

        for layer, af in zip(self.layers, self.activation_functions):
            if af:
                state_tensor = layer(state_tensor)
                state_tensor = af(state_tensor)
            else:
                state_tensor = layer(state_tensor)

        return state_tensor

    def backward(self, targets, states):
        # Training process, set gradients to 0
        self.optimizer.zero_grad()

        # Compute output of the network given the states batch
        pred = self.forward(states)

        # Compute loss function
        loss = nn.functional.mse_loss(pred, targets)

        # Compute gradient
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)

        # Perform backward pass (backpropagation)
        self.optimizer.step()


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ExperienceReplayBuffer(object):

    def __init__(self, maximum_length):
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n):
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        indices = np.random.choice(len(self.buffer), size=n, replace=False)

        batch = [self.buffer[i] for i in indices]

        # convert a list of tuples into a tuple of list we do zip(*batch)
        return zip(*batch)
