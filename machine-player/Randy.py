# This class inherits from Agent
# Randy is a one Agent who is able to play Catan
# He always picks randomly from the set of legal actions
# He is not able to learn

# Import the Agent class
from Agent import Agent

# Define the Randy class
class Randy(Agent):
    def __init__(self):
        # Set the exploration rate to 1
        self.exploration_rate = 1
        # Set the man's name, go on
        self.name = "Randy"
