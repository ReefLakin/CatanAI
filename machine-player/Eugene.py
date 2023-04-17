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
from LSTMModel import LSTMCatanModel

# Import the torch library
import torch

# Some settings
NORMALISE_STATES = False
LEGAL_ACTIONS_ONLY = True

# Define the Adam class
class Eugene(Agent):
    def __init__(self, exploration_rate=1):
        super().__init__(exploration_rate)
        self.name = "Eugene"
        self.model = LSTMCatanModel()
        self.hidden = self.model.init_hidden()

    # In this version of Adam's action selecting, only legal actions are considered
    def select_action_exploit(self, observation, all_possible_actions, legal_actions):

        # Preprocessing
        observation_processed = self.preprocess_state(observation)

        # Normalising
        if NORMALISE_STATES:
            observation_processed = self.normalise_state(observation_processed)

        # Tensor conversion
        observation_processed = torch.tensor(observation_processed, dtype=torch.float32)

        # Forward pass
        action_options, hidden = self.model.forward(observation_processed, self.hidden)

        # Update hidden state
        self.hidden = hidden

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
        loss = self.model.learn(self.memory, batch_size=32)
        return loss

    # Reward function
    def reward(self, reward_information):
        # Adam gets rewarded more for: reaching 10 VPs
        # Adam gets moderately rewarded for: building roads, settlements, cities
        # Adam gets mildly punished for: making illegal moves
        legal_actions = reward_information["legal_actions"]
        done = reward_information["game_over"]
        action = reward_information["current_action"]

        if done == True:
            return 10  # Victory reward
        elif action not in legal_actions:
            return 0  # Illegal move punishment
        else:
            split_action = action.split("_")
            if split_action[0] == "build" and split_action[1] == "road":
                return 0  # Road building reward
            elif split_action[0] == "build" and split_action[1] == "settlement":
                return 1  # Settlement building reward
            elif split_action[0] == "build" and split_action[1] == "city":
                return 1  # City building reward
            else:
                return 0  # Else, no reward
