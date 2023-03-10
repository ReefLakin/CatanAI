# This class inherits from Agent
# Adam is a one Agent who is able to play Catan
# He is the first Agent I've created who is able to learn
# His name is inspired by the Adam optimizer, but also the story of Creation in the Bible
# Perhaps a bit cliché, but I like it
# Adam can select any action, even if it is not legal
# He is able to learn from his mistakes
# By default, he will explore 10% of the time

# Import the Agent class
from Agent import Agent

# Import the CatanModel class
from Model import CatanModel

# Import the torch library
import torch

# Define the Adam class
class Adam(Agent):
    def __init__(self, exploration_rate=0.1):
        # Set the exploration rate to the exploration rate passed to the constructor
        self.exploration_rate = exploration_rate
        # Set the model (use the default values for now)
        self.model = CatanModel()
        # Name this fool
        self.name = "Adam"

    # Method for selecting an action via exploitation
    def select_action_exploit(self, observation, all_possible_actions, legal_actions):
        # Preprocess the state information
        observation_processed = self.preprocess_state(observation)
        # Convert the observation to a tensor
        observation_processed = torch.tensor(observation_processed, dtype=torch.float32)
        # Pass the observation through the model
        action_options = self.model.forward(observation_processed)
        # Acquire the singular action with the highest value
        action_as_idx = torch.argmax(action_options).item()
        # Get the actual action from the action index
        action = all_possible_actions[action_as_idx]
        # Return the action
        return action

    # Method for learning
    def learn(self, memory):
        self.model.learn(memory, batch_size=32)

    # Reward function
    def reward(self, reward_information):
        # Adam gets rewarded more for: reaching 10 VPs
        # Adam gets moderately rewarded for: building roads, settlements, cities
        # Adam gets mildly punished for: making illegal moves
        legal_actions = reward_information["legal_actions"]
        done = reward_information["game_over"]
        action = reward_information["current_action"]

        if done == True:
            return 100  # Victory reward
        elif action not in legal_actions:
            return -2  # Illegal move punishment
        else:
            split_action = action.split("_")
            if split_action[0] == "build" and split_action[1] == "road":
                return 1  # Road building reward
            elif split_action[0] == "build" and split_action[1] == "settlement":
                return 10  # Settlement building reward
            elif split_action[0] == "build" and split_action[1] == "city":
                return 10  # City building reward
            else:
                return 0  # Else, no reward
