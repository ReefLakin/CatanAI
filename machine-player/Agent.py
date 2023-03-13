# Import the random library
import random

# Import the StatePreprocessor class
from StatePreprocessor import StatePreprocessor

# Define the Agent class
class Agent:

    # Method for selecting an action
    def select_action(self, observation, all_possible_actions, legal_actions):
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
        # Define a new state preprocessor
        state_preprocessor = StatePreprocessor()
        # Preprocess the state
        new_state_list = state_preprocessor.preprocess_state(state)
        # Return the new state list
        return new_state_list

    # Method for learning
    # Will be overwritten by child classes, so let's just pass
    def learn(self, memory):
        pass

    # Method for loading the model
    def load_model(self, path):
        self.model.load(path)

    # Method for saving the model
    def save_model(self, path):
        self.model.save(path)

    # Reward function
    def reward(self, reward_information):
        return 0

    # Setter for the exploration rate
    def set_exploration_rate(self, exploration_rate):
        self.exploration_rate = exploration_rate
