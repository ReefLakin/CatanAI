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
    def __init__(self, exploration_rate=1):
        super().__init__(exploration_rate)
        self.name = "Adam"

    # In this version of Adam's action selecting, only legal actions are considered
    def select_action_exploit(self, observation, all_possible_actions, legal_actions):
        # Preprocess the state information
        observation_processed = self.preprocess_state(observation)
        # # Normalise the states in the states list
        # observation_processed = self.normalise_state(observation_processed)
        # # Convert the observation to a tensor
        # observation_processed = torch.tensor(observation_processed, dtype=torch.float32)
        # # Pass the observation through the model
        # action_options = self.model.forward(observation_processed)
        # # Acquire the singular action with the highest value
        # action_as_idx = torch.argmax(action_options).item()
        # # Get the actual action from the action index
        # action = all_possible_actions[action_as_idx]
        # # Return the action
        # return action

        # Convert the observation to a tensor
        observation_processed = torch.tensor(observation_processed, dtype=torch.float32)
        # Pass the observation through the model
        action_options = self.model.forward(observation_processed)
        # Get the indices of legal actions
        legal_action_indices = [
            i
            for i in range(len(all_possible_actions))
            if all_possible_actions[i] in legal_actions
        ]
        # Convert action_options from a tensor to a list
        action_options = action_options.tolist()
        # Loop through all the action options
        # If the action is not legal, set its value to -50
        # Otherwise, leave it alone
        for i in range(len(action_options)):
            if i not in legal_action_indices:
                action_options[i] = -50
        # Get the index of the highest value in the action options list
        action_as_idx = action_options.index(max(action_options))
        # Get the actual action from the action index
        action = all_possible_actions[action_as_idx]
        # Return the legal action
        return action

    # Method for learning
    def learn(self):
        self.model.learn(self.memory, batch_size=32)

    # Reward function
    def reward(self, reward_information):
        # Adam gets rewarded more for: reaching 10 VPs
        # Adam gets moderately rewarded for: building roads, settlements, cities
        # Adam gets mildly punished for: making illegal moves
        legal_actions = reward_information["legal_actions"]
        done = reward_information["game_over"]
        action = reward_information["current_action"]

        if done == True:
            return 80  # Victory reward
        elif action not in legal_actions:
            return -1  # Illegal move punishment
        else:
            split_action = action.split("_")
            if split_action[0] == "build" and split_action[1] == "road":
                return 10  # Road building reward
            elif split_action[0] == "build" and split_action[1] == "settlement":
                return 30  # Settlement building reward
            elif split_action[0] == "build" and split_action[1] == "city":
                return 35  # City building reward
            else:
                return 0  # Else, no reward
