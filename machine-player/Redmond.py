# Redmond inherits from the Agent class
# He is quite similar to Adam, but he has a different reward function
# He is rewarded more for building settlements and cities on red tiles (6 and 8)

# Imports
from Agent import Agent
import torch

# High-level settings
NORMALISE_STATES = True
LEGAL_ACTIONS_ONLY = True


# Class definition
class Redmond(Agent):
    def __init__(self, exploration_rate=1.0):
        super().__init__(exploration_rate)
        self.name = "Redmond"

    # Overwrite the select_action_exploit method with Redmond's own
    def select_action_exploit(self, observation, all_possible_actions, legal_actions):

        # State preprocessing
        observation_processed = self.preprocess_state(observation)

        # Normalising
        if NORMALISE_STATES:
            observation_processed = self.normalise_state(observation_processed)

        # Tensor conversion
        observation_processed = torch.tensor(observation_processed, dtype=torch.float32)

        # Forward pass
        action_options = self.model.forward(observation_processed)

        # Redmond has the option to only choose legal actions
        if LEGAL_ACTIONS_ONLY:
            # Create a list of legal action indices
            legal_action_indices = []
            for i in range(len(all_possible_actions)):
                if all_possible_actions[i] in legal_actions:
                    legal_action_indices.append(i)

            # Acquire the legal action with the highest value
            # Skip over the illegal actions
            current_best = -50
            for i in range(len(action_options)):
                if i in legal_action_indices:
                    if action_options[i] > current_best:
                        action_as_idx = i
                        current_best = action_options[i]

            # Get the actual action from the action index
            action = all_possible_actions[action_as_idx]

        else:
            # Acquire the singular action with the highest value
            action_as_idx = torch.argmax(action_options).item()
            # Get the actual action from the action index
            action = all_possible_actions[action_as_idx]

        # Action return
        return action

    # Method for learning
    def learn(self):
        self.model.learn(self.memory, batch_size=128)

    # Reward function
    def reward(self, reward_information):

        # These reward values tend tend to get adjusted during training
        reward_victory = 1
        reward_illegal_move = 0
        reward_road_building = 0.01
        reward_settlement_building = 0.05
        reward_city_building = 0.05
        reward_other = 0
        reward_red_tile = 0.3

        # Extract information from the reward information dictionary
        legal_actions = reward_information["legal_actions"]
        resource_bonus = reward_information["recent_resources_generated"]
        action = reward_information["current_action"]
        red_tiles = reward_information["red_tiles"]
        done = reward_information["game_over"]

        split_action_parts = action.split("_")

        # Reward assignment
        if done == True:
            return reward_victory
        elif action not in legal_actions:
            return reward_illegal_move
        else:
            # Settlement building reward
            if (
                split_action_parts[0] == "build"
                and split_action_parts[1] == "settlement"
            ):
                for tile in red_tiles:
                    q = tile[0]
                    r = tile[1]
                    s = tile[2]
                    if (
                        split_action_parts[3] == str(q)
                        and split_action_parts[4] == str(r)
                        and split_action_parts[5] == str(s)
                    ):
                        return reward_red_tile  # Red bonus reward

                return reward_settlement_building  # Non-red reward
            # City building reward
            elif split_action_parts[0] == "build" and split_action_parts[1] == "city":
                for tile in red_tiles:
                    q = tile[0]
                    r = tile[1]
                    s = tile[2]
                    if (
                        split_action_parts[3] == str(q)
                        and split_action_parts[4] == str(r)
                        and split_action_parts[5] == str(s)
                    ):
                        return reward_red_tile  # Red bonus reward

                return reward_city_building  # Non-red reward
            elif split_action_parts[0] == "build" and split_action_parts[1] == "road":
                return reward_road_building

        # Other reward
        return reward_other
