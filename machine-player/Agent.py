"""
The class will contain the following methods:
- learn (for receiving information on the current state of the game)
- preprocess (for preprocessing the game state in a format that the model understands best)
- select_action (for choosing which action to take)

The class will contain the following attributes:
- model (which is the model that the agent will use to learn and make decisions)
- replay_memory (which is the replay memory that the agent will use to store experiences)
- policy (which is the policy that the agent will use to select actions)
- learning_rate (which is an adjustable hyperparameter used to fine-tune the Agent)
- discount_factor (which is an adjustable hyperparameter used to fine-tune the Agent)
"""

class Agent:

    def __init__(self, model, replay_memory, policy, learning_rate, discount_factor):
        self.model = model
        self.replay_memory = replay_memory
        self.policy = policy
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def learn(self, new_state, reward, game_over):
        # Make an observation of the game state
        pass

    def preprocess(self, batch):
        # Preprocess a batch of state information and return it
        pass

    def select_action(self, state):
        # Select an action to take based on the given state and the Agent's policy
        pass