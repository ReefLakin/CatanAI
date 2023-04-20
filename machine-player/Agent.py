# This is the Agent base class that all other agents inherit from
# It has a few components that all agents will share
# But these can be overwritten by child classes if necessary

# Imports
import random
from StatePreprocessor import StatePreprocessor
import numpy as np
from ReplayMemory import ReplayMemory
from Model import CatanModel


# Agent class definition
class Agent:
    def __init__(self, exploration_rate=0.1):
        self.exploration_rate = exploration_rate
        self.model = CatanModel()
        self.name = "Agent"
        self.memory = ReplayMemory(100000)

    # Method for selecting an action
    def select_action(self, observation, all_possible_actions, legal_actions):
        # This line can be disabled; prints the legal actions (takes up a lot of console space)
        print(legal_actions)

        # If the random number is less than the exploration rate, choose a random action
        if random.random() < self.exploration_rate:
            action = random.choice(legal_actions)

        # Otherwise, use the model to predict the best action
        else:
            action = self.select_action_exploit(
                observation, all_possible_actions, legal_actions
            )

        # Return the action
        return action

    # Method for selecting an action via exploitation
    # Will be overwritten by child classes, but let's return a random legal action anyway
    def select_action_exploit(self, observation, all_possible_actions, legal_actions):
        return random.choice(legal_actions)

    # Preprocess the state information passed to the select_action method
    def preprocess_state(self, state):
        state_preprocessor = StatePreprocessor()
        new_state_list = state_preprocessor.preprocess_state(state)
        return new_state_list

    # Method for learning
    # Will be overwritten by child classes, so let's just pass
    def learn(self):
        pass

    # Method for loading the model
    def load_model(self, path):
        self.model.load(path)

    # Method for saving the model
    def save_model(self, path):
        self.model.save(path)

    # Reward function
    # Will be overwritten by child classes, hopefully
    def reward(self, reward_information):
        return 0

    # Setter for the exploration rate
    def set_exploration_rate(self, exploration_rate):
        self.exploration_rate = exploration_rate

    # Getter for the exploration rate
    def get_exploration_rate(self):
        return self.exploration_rate

    # Method for normalising the state (not really used as much)
    def normalise_state(self, state):
        return (state - np.min(state)) / (np.max(state) - np.min(state))

    # Method for feeding the agent memory
    def feed_memory(self, observation):
        self.memory.add(observation)
