# Brain for the machine player
# Consists of a policy network and a target network
# Also contains the functions for training the policy network

# Imports
import torch.nn as nn
import torch
from DQN import DQN
from StatePreprocessor import StatePreprocessor
import numpy as np

DEFAULT_INPUT_SIZE = 272
DEFAULT_OUTPUT_SIZE = 382
DEFAULT_HIDDEN_SIZE = 42


# Class definition
class Brain:
    def __init__(
        self,
        input_size=DEFAULT_INPUT_SIZE,
        output_size=DEFAULT_OUTPUT_SIZE,
        hidden_size=DEFAULT_HIDDEN_SIZE,
    ):
        # Initialise the policy network
        self.policy_network = DQN(input_size, output_size, hidden_size)

        # Initialise the target network
        self.target_network = DQN(input_size, output_size, hidden_size)

        # Set the weights and biases of the target network to be the same as the policy network
        self.target_network.load_state_dict(self.policy_network.state_dict())

        # Set the target network to evaluation mode
        self.target_network.eval()

        # Set the target network to not require gradients
        self.target_network.requires_grad_(False)

        # Set the policy network to training mode
        self.policy_network.train()

        # Set the policy network to require gradients
        self.policy_network.requires_grad_(True)

        # Set the target update counter to zero
        self.target_update_counter = 0

        # Set the target update frequency to 1000
        self.target_update_frequency = 500

    def update_target_network(self):
        # Set the weights and biases of the target network to be the same as the policy network
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def train_policy_network(self, memory, batch_size=32, gamma=0.99):
        # If the buffer if not full enough, return
        if memory.get_buffer_size() < batch_size:
            return

        # Sample from the replay memory
        # This is a list of tuples in the format (state, action, reward, next_state)
        transitions = memory.sample(batch_size)

        # Convert the list of tuples into separate lists
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Create a new state preprocessor
        state_preprocessor = StatePreprocessor()

        # Preprocess the states in the states list
        states = [state_preprocessor.preprocess_state(state) for state in states]

        # Preprocess the next states in the next_states list
        next_states = [
            state_preprocessor.preprocess_state(state) for state in next_states
        ]

        # Normalise the states in the states list
        states = [self.normalise_state(state) for state in states]

        # Normalise the next states in the next_states list
        next_states = [self.normalise_state(state) for state in next_states]

        # Convert the lists into tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.long)

        # Get the Q values for the current states
        q_values = self.policy_network.forward(states)

        # Get the Q values for the actions taken
        q_values_for_actions_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get the Q values for the next states
        next_q_values = self.target_network.forward(next_states)

        # Get the maximum Q values for the next states
        max_next_q_values = torch.max(next_q_values, dim=1)[0]

        # Calculate the target Q values
        target_q_values = rewards + (1 - dones) * gamma * max_next_q_values

        # Calculate the loss
        loss = nn.SmoothL1Loss()(q_values_for_actions_taken, target_q_values)

        # Zero the gradients
        self.policy_network.optimiser.zero_grad()

        # Backpropagate the loss
        loss.backward()

        # Clip the gradients (prevents exploding gradients)
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)

        # Update the weights and biases
        self.policy_network.optimiser.step()

        # Increment the target update counter
        self.target_update_counter += 1

        # If the target update counter is greater than the target update frequency
        if self.target_update_counter > self.target_update_frequency:
            # Update the target network
            self.update_target_network()

            # Reset the target update counter
            self.target_update_counter = 0

        return loss.item()

    # Function for selecting the best action
    def select_best_action(self, state):
        with torch.no_grad():
            return self.policy_network(state).max(1)[1].view(1, 1)

    # Function for normalising a state
    def normalise_state(self, state):
        # Normalise the state
        state = (state - np.min(state)) / (np.max(state) - np.min(state))
        return state

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        print("Loading model from file: " + filename)
        self.load_state_dict(torch.load(filename))
