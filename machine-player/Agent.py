# Import the random library
import random

# Import the torch library
import torch

# Import the time library
import time

# Import the StatePreprocessor class
from StatePreprocessor import StatePreprocessor

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
    def select_action(self, observation, all_possible_actions, legal_actions):
        print(legal_actions)
        # If the random number is less than the exploration rate, choose a random action
        if random.random() < self.exploration_rate:
            action = random.choice(legal_actions)

        # Otherwise, use the model to predict the best action
        else:
            # Preprocess the state information
            observation_processed = self.preprocess_state(observation)
            # Convert the observation to a tensor
            observation_processed = torch.tensor(
                observation_processed, dtype=torch.float32
            )
            # Pass the observation through the model
            action_options = self.model.forward(observation_processed)
            # Acquire the singular action with the highest value
            action_as_idx = torch.argmax(action_options).item()
            # Get the actual action from the action index
            action = all_possible_actions[action_as_idx]

        # Return the action
        return action

    # Preprocess the state information passed to the select_action method
    def preprocess_state(self, state):
        # Define a new state preprocessor
        state_preprocessor = StatePreprocessor()
        # Preprocess the state
        new_state_list = state_preprocessor.preprocess_state(state)
        # Return the new state list
        return new_state_list

    # Method for learning
    def learn(self, memory):
        self.model.learn(memory)
