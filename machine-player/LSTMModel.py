# Imports
import torch.nn as nn
import torch
from StatePreprocessor import StatePreprocessor
import numpy as np
import random


class LSTMCatanModel(nn.Module):
    def __init__(self, input_size=525, output_size=382, hidden_size=64):
        super(LSTMCatanModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

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

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state with zeros
        return (
            torch.zeros(1, batch_size, self.hidden_size),
            torch.zeros(1, batch_size, self.hidden_size),
        )

    def learn(self, memory, batch_size=128, gamma=0.99, sequence_length=8):
        # If the buffer is not full enough, return
        if memory.get_buffer_size() < batch_size:
            return

        # Zero the gradients
        self.optimiser.zero_grad()

        # Sample from the replay memory
        transitions = memory.sample(batch_size)

        # Convert the list of transitions into a batch of sequences
        # (batch_size, sequence_length, state_size)
        sequences = []
        for i in range(batch_size):
            start_index = random.randint(0, len(transitions) - sequence_length)
            sequence = transitions[start_index : start_index + sequence_length]
            sequence = [t[0] for t in sequence]  # Extract states from transitions
            sequences.append(sequence)
        sequences = np.array(sequences)

        # Convert the sequences into tensors
        sequences = torch.tensor(sequences, dtype=torch.float32)

        # Get the Q-values for the current sequence
        # (batch_size, sequence_length, output_size)
        q_values = self.forward(sequences)

        # Calculate the target Q-values
        # (batch_size, sequence_length, output_size)
        targets = torch.zeros_like(q_values)
        for i in range(batch_size):
            for j in range(sequence_length):
                state, action, reward, next_state, done = transitions[
                    i * sequence_length + j
                ]
                targets[i, j, action] = reward
                if not done:
                    targets[i, j, action] += gamma * torch.max(q_values[i, j + 1, :])

        # Flatten the target and predicted Q-values for computing the loss
        # (batch_size*sequence_length, output_size)
        targets = targets.view(-1, self.output_size)
        q_values = q_values.view(-1, self.output_size)

        # Compute the loss
        loss = nn.MSELoss()(q_values, targets)

        # Backpropagate the loss
        loss.backward()

        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

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
