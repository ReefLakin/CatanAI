# Import the random library
import random

# Define the Agent class
class Agent:
    def __init__(self, exploration_rate):
        self.exploration_rate = exploration_rate

    # Method for selecting an action
    def select_action(self, legal_actions):
        print(legal_actions)
        if random.random() < self.exploration_rate:
            return random.choice(legal_actions)

        # Eventually to be replaced with a call to the model (at the moment exp. rate is going to always be 1)
        else:
            return random.choice(legal_actions)
