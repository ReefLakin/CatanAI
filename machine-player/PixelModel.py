# Imports
import torch.nn as nn
import torch
from StatePreprocessor import StatePreprocessor
from PixelPreprocessor import PixelPreprocessor
import numpy as np


class CatanPixelModel(nn.Module):
    def __init__(self, output_size=382):
        super(CatanPixelModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 220 * 175, 128)
        self.fc2 = nn.Linear(128, 10)

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
        self.optimiser = torch.optim.AdamW(self.parameters(), lr=0.005)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 220 * 175)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
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
        pixels, actions, rewards, next_pixels, dones = zip(*transitions)

        # Create a new pixel preprocessor
        pixel_preprocessor = PixelPreprocessor()

        # Preprocess the states in the states list
        pixels_preprocessed = [
            pixel_preprocessor.normalise_pixel_array(pixel) for pixel in pixels
        ]

        # Preprocess the next states in the next_states list
        next_pixels_preprocessed = [
            pixel_preprocessor.normalise_pixel_array(pixel) for pixel in next_pixels
        ]

        # Numpy ndarrays are slow to convert to tensors, so convert them to normal np arrays first
        pixels_preprocessed = np.array(pixels_preprocessed, dtype=np.float32)
        next_pixels_preprocessed = np.array(next_pixels_preprocessed, dtype=np.float32)

        # Convert the lists into tensors
        pixels_preprocessed = torch.tensor(pixels_preprocessed, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_pixels_preprocessed = torch.tensor(
            next_pixels_preprocessed, dtype=torch.float32
        )
        dones = torch.tensor(dones, dtype=torch.long)

        # Get the Q values for the current states
        q_values = self.forward(pixels_preprocessed)

        # Get the Q values for the actions taken
        q_values_for_actions_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get the Q values for the next states
        next_q_values = self.forward(next_pixels_preprocessed)

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

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        print("Loading model from file: " + filename)
        self.load_state_dict(torch.load(filename))
