import numpy as np
import gym
from collections import deque
from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F

replay_buffer = deque(maxlen=1000)


class Net(nn.Module):

    def __init__(self, states, actions):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(states, 8)
        self.fc2 = nn.Linear(8, actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


### CREATE RL ENVIRONMENT ###
env = gym.make('CartPole-v0')  # Create a CartPole environment
n = len(env.observation_space.low)  # State space dimensionality
m = env.action_space.n  # Number of actions

nn = Net(n, m)
optim = torch.optim.Adam(nn.parameters(), lr=0.01)

### PLAY ENVIRONMENT ###
# The next while loop plays 5 episode of the environment
for episode in range(50):
    state = env.reset()  # Reset environment, returns initial state
    done = False  # Boolean variable used to indicate if an episode terminated

    while not done:
        env.render()  # Render the environment (DO NOT USE during training of the labs...)
        # action  = np.random.randint(m)   # Pick a random integer between [0, m-1]
        state_tensor = torch.tensor([state], requires_grad=False)
        action_tensor = nn.forward(state_tensor)
        action = action_tensor.max(1)[1].item()
        # print(f"state {state_tensor} -> actions {action_tensor} -> action {action}")

        # The next line takes permits you to take an action in the RL environment
        # env.step(action) returns 4 variables:
        # (1) next state; (2) reward; (3) done variable; (4) additional stuff
        next_state, reward, done, *_ = env.step(action)

        replay = (state, action, reward, next_state, done)
        replay_buffer.append(replay)

        state = next_state


        batch_size = 3
        if len(replay_buffer) >= batch_size:
            # set gradients to 0
            optim.zero_grad()

            # get batch for training
            batch = sample(list(replay_buffer), batch_size)
            batch = list(zip(*batch))

            # get states from batch
            batch_states = np.array(batch[0])
            states = torch.tensor(batch_states, requires_grad=False)

            # run NN for action prediction
            action_predictions = nn(states)

            # zero vector like prediction - why??
            target = torch.zeros_like(action_predictions, requires_grad=False)

            # compute loss
            loss = torch.nn.functional.mse_loss(action_predictions, target)

            # compute gradients
            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(nn.parameters(), 1.)

            # perform backprop
            optim.step()


# Close all the windows
env.close()
