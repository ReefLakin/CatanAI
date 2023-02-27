# Import the random library
import random

# Import the torch library
import torch

# Import the CatanModel class
from Model import CatanModel

# Define the Agent class
class Agent:
    def __init__(self, exploration_rate):
        # Set the exploration rate
        self.exploration_rate = exploration_rate
        # Set the model (use the default values for now)
        self.model = CatanModel()

    # Method for selecting an action
    def select_action(self, observation, all_possible_actions, legal_actions):
        print(legal_actions)
        # If the random number is less than the exploration rate, choose a random action
        if random.random() < self.exploration_rate:
            action = random.choice(legal_actions)

        # Otherwise, use the model to predict the best action
        else:
            action_selected_is_legal = False
            while not action_selected_is_legal:
                # Preprocess the state information
                observation_processed = self.preprocess_state(observation)
                # Convert the observation to a tensor
                observation_processed = torch.tensor(
                    observation_processed, dtype=torch.float32
                )
                # Pass the observation through the model
                action_options = self.model.forward(observation_processed)
                # Acquire the singular action with the highest value
                action = torch.argmax(action_options).item()
                # Get the actual action from the action index
                action = all_possible_actions[action]
                # Check if the action is legal
                if action in legal_actions:
                    action_selected_is_legal = True

                # Print the action
                print(action)
                # Request user input before continuing
                input()

        # Return the action
        return action

    # Preprocess the state information passed to the select_action method
    def preprocess_state(self, state):
        new_state_list = []

        # Victory points
        new_state_list.append(state["victory_points"])

        # Turn number
        new_state_list.append(state["turn_number"])

        # Total number of ore
        new_state_list.append(state["num_ore"])

        # Total number of grain
        new_state_list.append(state["num_grain"])

        # Total number of wool
        new_state_list.append(state["num_wool"])

        # Total number of lumber
        new_state_list.append(state["num_lumber"])

        # Total number of brick
        new_state_list.append(state["num_brick"])

        # Loop over each tile for each side
        for tile in state["side_states"]:
            # Loop over each side
            for side in tile:
                # Add the side to the new state list
                new_state_list.append(side)

        # Loop over each tile for each vertex
        for tile in state["vertex_states"]:
            # Loop over each vertex
            for vertex in tile:
                # Add the vertex to the new state list
                new_state_list.append(vertex)

        # Loop over tile types
        for tile_type in state["tile_types"]:
            # Change string values to integers
            # brick = 1, lumber = 2, wool = 3, grain = 4, ore = 5, desert = 6
            if tile_type == "brick":
                new_state_list.append(1)
            elif tile_type == "lumber":
                new_state_list.append(2)
            elif tile_type == "wool":
                new_state_list.append(3)
            elif tile_type == "grain":
                new_state_list.append(4)
            elif tile_type == "ore":
                new_state_list.append(5)
            elif tile_type == "desert":
                new_state_list.append(6)

        # Loop over tile numbers
        for tile_number in state["tile_values"]:
            # Add the tile number to the new state list
            new_state_list.append(tile_number)

        # Loop over the board dimensions
        for row in state["board_dims"]:
            # Add the board dimension to the new state list
            new_state_list.append(row)

        # Return the new state list where None values are replaced with 0 (through list comprehension)
        return [0 if v is None else v for v in new_state_list]
