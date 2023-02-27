# Import the random library
import random

# Import the torch library
import torch

# Import the CatanModel class
from Model import CatanModel

# Define the Agent class
class Agent:
    def __init__(self, exploration_rate):
        # Set the exploration rate
        self.exploration_rate = exploration_rate
        # Set the model (use the default values for now)
        self.model = CatanModel()

    # Method for selecting an action
    def select_action(self, observation, legal_actions):
        print(legal_actions)
        # If the random number is less than the exploration rate, choose a random action
        if random.random() < self.exploration_rate:
            action = random.choice(legal_actions)

        # Otherwise, use the model to predict the best action
        else:
            # Convert the observation to a tensor
            observation = torch.tensor(observation, dtype=torch.float32)
            # Pass the observation through the model
            action_options = self.model.forward(observation)
            # Acquire the singular action with the highest value
            action = torch.argmax(action_options).item()

        # Return the action
        return action
