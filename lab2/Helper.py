# Load packages
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn


class StateActionValueNetwork(nn.Module):

    def __init__(self, dim_states: int, n_actions: int, hidden_layer_sizes: list, lr: float, weight_decay: float = 0):
        super(StateActionValueNetwork, self).__init__()
        self.dim_states = dim_states
        self.n_actions = n_actions
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = None
        self.activation_functions = None
        self.optimizer = None
        self.lr = lr
        self.max_grad_norm = 1  # advised range [0.5, 2]
        self.weight_decay = weight_decay

        self.setup_NN(dim_states, hidden_layer_sizes, n_actions)
        self.setup_optimizer()

    def setup_NN(self, dim_states, hidden_layer_sizes, n_actions, intermediate_af=nn.ReLU(), last_af=None):
        layers = nn.ModuleList()
        activation_functions = nn.ModuleList()

        layer_definitions = [dim_states] + hidden_layer_sizes
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, state: torch.Tensor):
        # out = torch.from_numpy(state)
        out = state

        for layer, af in zip(self.layers, self.activation_functions):
            if af:
                out = layer(out)
                out = af(out)
            else:
                out = layer(out)

        return out

    def backward(self, current, targets):
        # Training process, set gradients to 0
        self.optimizer.zero_grad()

        # Compute loss function
        loss = nn.functional.mse_loss(current, targets)

        # Compute gradient
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)

        # Perform backward pass (backpropagation)
        self.optimizer.step()

        return (loss / len(current)).detach().numpy()


class ActorPolicyNetwork(nn.Module):
    def __init__(self, dim_states: int, dim_actions: int, hidden_layer_sizes: list, lr: float, weight_decay: float = 0):
        super(ActorPolicyNetwork, self).__init__()
        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = None
        self.activation_functions = None
        self.optimizer = None
        self.lr = lr
        self.max_grad_norm = 1
        self.weight_decay = weight_decay

        self.setup_NN(dim_states, hidden_layer_sizes, dim_actions)
        self.setup_optimizer()

    def setup_NN(self, dim_states, hidden_layer_sizes, dim_actions, intermediate_af=nn.ReLU(), last_af=nn.Tanh()):
        layers = nn.ModuleList()
        activation_functions = nn.ModuleList()

        layer_definitions = [dim_states] + hidden_layer_sizes
        # layer_definitions.append(n_actions)

        for i in range(len(layer_definitions) - 1):
            layer = nn.Linear(layer_definitions[i], layer_definitions[i + 1])
            layers.append(layer)
            activation_functions.append(intermediate_af)

        # last layer with special
        layers.append(nn.Linear(layer_definitions[-1], dim_actions))
        activation_functions.append(last_af)

        self.layers = layers
        self.activation_functions = activation_functions

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, state: torch.Tensor):
        # out = torch.from_numpy(state)
        out = state

        for layer, af in zip(self.layers, self.activation_functions):
            if af:
                out = layer(out)
                out = af(out)
            else:
                out = layer(out)

        return out

    def backward(self, policy_loss: torch.Tensor):
        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()


class CriticValueNetwork(nn.Module):
    def __init__(self, dim_states: int, dim_actions: int, hidden_layer_sizes: list, lr: float,
                 weight_decay: float = 0):
        super(CriticValueNetwork, self).__init__()
        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = None
        self.activation_functions = None
        self.optimizer = None
        self.lr = lr
        self.max_grad_norm = 1
        self.weight_decay = weight_decay

        self.setup_NN(dim_states, hidden_layer_sizes, 1, dim_actions)
        self.setup_optimizer()

    def setup_NN(self, dim_states, hidden_layer_sizes, out_dim, dim_actions, intermediate_af=nn.ReLU(),
                 last_af=None):
        layers = nn.ModuleList()
        activation_functions = nn.ModuleList()

        layer_definitions = [dim_states] + hidden_layer_sizes

        for i in range(len(layer_definitions) - 1):
            if i == 1:
                # second layer gets additional input
                layer = nn.Linear(layer_definitions[i] + dim_actions, layer_definitions[i + 1])
            else:
                layer = nn.Linear(layer_definitions[i], layer_definitions[i + 1])
            layers.append(layer)
            activation_functions.append(intermediate_af)

        # last layer with special
        layers.append(nn.Linear(layer_definitions[-1], out_dim))
        activation_functions.append(last_af)

        self.layers = layers
        self.activation_functions = activation_functions

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        # out = torch.from_numpy(state)
        out = state

        for i, (layer, af) in enumerate(zip(self.layers, self.activation_functions)):
            if i == 1:
                # concat output of input layer with action
                out = torch.cat([out, action], dim=1)

            if af:
                out = layer(out)
                out = af(out)
            else:
                out = layer(out)

        return out

    def backward(self, Q_vals: torch.Tensor, Q_targets: torch.Tensor):
        self.optimizer.zero_grad()
        loss = nn.functional.mse_loss(Q_vals, Q_targets)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ExperienceReplayBuffer(object):

    def __init__(self, maximum_length):
        self.buffer = deque(maxlen=int(maximum_length))

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n, combined=True):
        n = int(n)
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        if combined:
            n -= 1  # fetch 1 less
            indices = np.random.choice(len(self.buffer) - 1, size=n, replace=False)  # consider only rest of experiences
            batch = [self.buffer[i] for i in indices]
            batch.append(self.buffer[-1])  # append newest experience
        else:
            indices = np.random.choice(len(self.buffer), size=n, replace=False)
            batch = [self.buffer[i] for i in indices]

        # convert a list of tuples into a tuple of list we do zip(*batch)
        return zip(*batch)
