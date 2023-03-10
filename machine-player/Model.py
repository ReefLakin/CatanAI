# Imports
import torch.nn as nn
import torch
from StatePreprocessor import StatePreprocessor
import numpy as np


class CatanModel(nn.Module):
    def __init__(self, input_size=278, output_size=363, hidden_size=64):
        super(CatanModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Initialise weights with the He uniform method
        # Loop through all layers and initialise the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # a represents the slope of the negative part of the rectifier, default is 0
                # mode is the method of initialisation
                # nonlinearity is the activation function
                nn.init.kaiming_uniform_(
                    m.weight, a=0, mode="fan_in", nonlinearity="relu"
                )
                # Initialise biases to zero, which is the default anyway
                nn.init.zeros_(m.bias)

        # Initialise the optimiser
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def learn(self, memory, batch_size=16, gamma=0.99):
        # Sample from the replay memory
        # This is a list of tuples in the format (state, action, reward, next_state)
        transitions = memory.sample(batch_size)

        # Convert the list of tuples into separate lists
        states, actions, rewards, next_states = zip(*transitions)

        # Create a new state preprocessor
        state_preprocessor = StatePreprocessor()

        # Preprocess the states in the states list
        states = [state_preprocessor.preprocess_state(state) for state in states]

        # Preprocess the next states in the next_states list
        next_states = [
            state_preprocessor.preprocess_state(state) for state in next_states
        ]

        # Normalise the states in the states list
        states = [self.normalise_state(state) for state in states]

        # Normalise the next states in the next_states list
        next_states = [self.normalise_state(state) for state in next_states]

        # Convert the lists into tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        # Calculate the q values using the states and actions provided
        # The use of gather() here is essentially a glorified for loop; it's a more efficient way of calculating the q values
        q_values = self(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Use the Bellman equation to calculate the target q values
        # reward: the tensor representing the reward for each transition
        # gamma: the discount factor
        # next_state: the tensor representing the next state for each transition
        # max(1)[0]: the maximum q value for each transition
        # detach(): detach the tensor from the computational graph
        # targets: the target q values, as calculated by the Bellman equation on this line
        targets = rewards + gamma * self(next_states).max(1)[0].detach()

        # Calculate the loss
        # The loss is the mean squared error between the q values and the target q values
        loss = nn.MSELoss()(q_values, targets)

        # Zero the gradients
        self.optimiser.zero_grad()

        # Calculate the gradients
        loss.backward()

        # Update the weights
        self.optimiser.step()

    def normalise_state(self, state):
        # Normalise the state
        return (state - np.min(state)) / (np.max(state) - np.min(state))

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        print("Loading model from file: " + filename)
        self.load_state_dict(torch.load(filename))
