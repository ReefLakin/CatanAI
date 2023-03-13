# This class inherits from Agent
# Redmond is my second Agent who is able to play Catan
# He is a little more experimental than Adam
# His name is inspired by my friend, Alex Redmond, but also the idea that he loves building near red tiles (6 and 8)
# Like Adam, Redmon can select any action, even if it is not legal
# He SHOULD be able to learn from his mistakes
# Redmond is NOT rewarded for building roads
# He is rewarded for genering resources

# Import the Agent class
from Agent import Agent

# Import the CatanModel class
from Model import CatanModel

# Import the torch library
import torch

# Define the Redmond class
class Redmond(Agent):
    def __init__(self, exploration_rate=0.1):
        # Set the exploration rate to the exploration rate passed to the constructor
        self.exploration_rate = exploration_rate
        # Set the model (use the default values for now)
        self.model = CatanModel()
        # Name this lovely gentleman
        self.name = "Redmond"

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
        # Redmond gets rewarded for: building a settlement or city on red tiles (6 and 8)
        # Redmond gets a small reward for: generating resources
        # Redmond gets slightly less reward for: building settlements and cities on non-red tiles
        # Remond is not rewarded for: winning the game, or building roads
        # Remond is punished for: making illegal moves
        legal_actions = reward_information["legal_actions"]
        resource_bonus = reward_information["recent_recources_generated"]
        action = reward_information["current_action"]
        red_tiles = reward_information["red_tiles"]

        split_action_parts = action.split("_")

        # Illegal move penalty
        if action not in legal_actions:
            return -1

        # Settlement building reward
        if split_action_parts[0] == "build" and split_action_parts[1] == "settlement":
            for tile in red_tiles:
                q = tile[0]
                r = tile[1]
                s = tile[2]
                if (
                    split_action_parts[3] == str(q)
                    and split_action_parts[4] == str(r)
                    and split_action_parts[5] == str(s)
                ):
                    return 8 + resource_bonus  # Red bonus reward

            return 2 + resource_bonus  # Non-red reward

        # City building reward
        if split_action_parts[0] == "build" and split_action_parts[1] == "city":
            for tile in red_tiles:
                q = tile[0]
                r = tile[1]
                s = tile[2]
                if (
                    split_action_parts[3] == str(q)
                    and split_action_parts[4] == str(r)
                    and split_action_parts[5] == str(s)
                ):
                    return 12 + resource_bonus  # Red bonus reward

            return 4 + resource_bonus  # Non-red reward
