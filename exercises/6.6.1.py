import torch
import torch.nn as nn

# Parameters of the problem , freely chosen
K = 1  # Bound on the Q value
A = 2  # Number of actions
d = 2  # State dimensionality


# Network class , with one hidden layer of neurons
class Network1(nn.Module):
    def __init__(self, d, K, A):
        super().__init__()
        hidden_neurons = 10
        # Number of hidden neurons
        # The input dimensionality of the network should be
        # equal to the dimensionality of the state
        self.input_state_layer = nn.Linear(d, hidden_neurons)
        # The output of the network should be equal to the number
        # of actions
        self.output_layer = nn.Linear(hidden_neurons, A)
        # Use the Tanh activation to bound the output between -1 and 1
        self.output_activation = nn.Tanh()
        self.K = K

    # given s compute Q(s,a) for every a
    def forward(self, s):  # Computation of Q(s,a)
        h1 = self.input_state_layer(s)
        h2 = self.output_layer(h1)
        # Multiply the output of the activation function
        # by K to get that the output is between -K and K
        return self.K * self.output_activation(h2)


# Example
net = Network1(d, K, A)  # Instance of the network
x = torch.tensor([[1.] * d])  # A batch of data of size 1xd
print("Input: {} - Output: {}".format(x.detach().numpy(), net(x).detach().numpy()))
