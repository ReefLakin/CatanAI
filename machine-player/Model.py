# Imports
import torch.nn as nn
import torch
from StatePreprocessor import StatePreprocessor
import numpy as np


class CatanModel(nn.Module):
    def __init__(self, input_size=525, output_size=382, hidden_size=32):
        super(CatanModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)

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
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.004)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

    def learn(self, memory, batch_size=128, gamma=0.99):
        # If the buffer if not full enough, return
        if memory.get_buffer_size() < batch_size:
            return

        # Zero the gradients
        self.optimiser.zero_grad()

        # Sample from the replay memory
        # This is a list of tuples in the format (state, action, reward, next_state)
        transitions = memory.sample(batch_size)

        # Convert the list of tuples into separate lists
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Create a new state preprocessor
        state_preprocessor = StatePreprocessor()

        # Preprocess the states in the states list
        states = [state_preprocessor.preprocess_state(state) for state in states]

        # Preprocess the next states in the next_states list
        next_states = [
            state_preprocessor.preprocess_state(state) for state in next_states
        ]

        # # Normalise the states in the states list
        # states = [self.normalise_state(state) for state in states]

        # # Normalise the next states in the next_states list
        # next_states = [self.normalise_state(state) for state in next_states]

        # Convert the lists into tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.long)

        # Get the Q values for the current states
        q_values = self.forward(states)

        # Get the Q values for the actions taken
        q_values_for_actions_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get the Q values for the next states
        next_q_values = self.forward(next_states)

        # Get the maximum Q values for the next states
        max_next_q_values = torch.max(next_q_values, dim=1)[0]

        # Calculate the target Q values
        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

        # Calculate the loss
        loss = nn.MSELoss()(q_values_for_actions_taken, target_q_values)

        # Backpropagate the loss
        loss.backward()

        # Update the weights
        self.optimiser.step()

        # Return the loss as a float
        return loss.item()

    def normalise_state(self, state):
        # Normalise the state
        return (state - np.min(state)) / (np.max(state) - np.min(state))

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        print("Loading model from file: " + filename)
        self.load_state_dict(torch.load(filename))
