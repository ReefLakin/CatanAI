# Davish information here

# Imports
from Agent import Agent
import torch
from Brain import Brain

# High-level settings
NORMALISE_STATES = True
LEGAL_ACTIONS_ONLY = True


# Class definition
class Davish(Agent):
    def __init__(self, exploration_rate=1.0):
        super().__init__(exploration_rate)
        self.name = "Davish"
        self.brain = Brain()

    # Overwrite the select_action_exploit method with Davish's own
    def select_action_exploit(self, observation, all_possible_actions, legal_actions):
        # State preprocessing
        observation_processed = self.preprocess_state(observation)

        # Normalising
        if NORMALISE_STATES:
            observation_processed = self.normalise_state(observation_processed)

        # Tensor conversion
        observation_processed = torch.tensor(observation_processed, dtype=torch.float32)

        # Davush has the option to only choose legal actions
        if LEGAL_ACTIONS_ONLY:
            # Forward pass
            with torch.no_grad():
                action_options = self.brain.policy_network(observation_processed)

            # Create a list of legal action indices
            legal_action_indices = []
            for i in range(len(all_possible_actions)):
                if all_possible_actions[i] in legal_actions:
                    legal_action_indices.append(i)

            # Acquire the legal action with the highest value
            # Skip over the illegal actions
            current_best = -50
            action_as_idx = 0
            for i in range(len(action_options)):
                if i in legal_action_indices:
                    if action_options[i] > current_best:
                        action_as_idx = i
                        current_best = action_options[i]

            # Get the actual action from the action index
            action = all_possible_actions[action_as_idx]

        else:
            action_index = self.brain.select_best_action(observation_processed)
            action = all_possible_actions[action_index]

        # Action return
        return action

    # Method for learning
    def learn(self):
        loss = self.brain.train_policy_network(self.memory)
        return loss

    # Reward function
    def reward(self, reward_information):
        # These reward values tend to get adjusted during training
        reward_victory = 1
        reward_illegal_move = 0
        reward_road_building = 0
        reward_settlement_building = 0
        reward_city_building = 0
        reward_other = 0

        # Extract information from the reward information dictionary
        legal_actions = reward_information["legal_actions"]
        done = reward_information["game_over"]
        action = reward_information["current_action"]

        # Reward assignment
        if done == True:
            return reward_victory
        elif action not in legal_actions:
            return reward_illegal_move
        else:
            split_action = action.split("_")
            if split_action[0] == "build" and split_action[1] == "road":
                return reward_road_building
            elif split_action[0] == "build" and split_action[1] == "settlement":
                return reward_settlement_building
            elif split_action[0] == "build" and split_action[1] == "city":
                return reward_city_building
            else:
                return reward_other
