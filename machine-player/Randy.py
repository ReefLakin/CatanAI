# This class inherits from Agent
# Randy is a one Agent who is able to play Catan
# He always picks randomly from the set of legal actions
# He is not able to learn

# Import the Agent class
from Agent import Agent


# Define the Randy class
class Randy(Agent):
    def __init__(self, exploration_rate=1):
        super().__init__(exploration_rate)
        self.name = "Randy"
        self.nickname = "Randy"

    def feed_memory(self, observation):
        return
