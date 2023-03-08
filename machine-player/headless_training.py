# path: machine-player/headless_training.py

# Imports

# OS
import os

# CATAN GAME
from CatanGame import CatanGame

# AGENTS
from Randy import Randy
from Adam import Adam

# REPLAY MEMORY
from ReplayMemory import ReplayMemory


# Set Constants

# BOARD DIMENSIONS
BOARD_DIMS = [3, 4, 5, 4, 3]

# THE ACTUAL BOARD
game_instance = CatanGame()

# THE AGENT
AGENT_SELECTED = "Adam"

# REPLAY MEMORY
replay_memory = ReplayMemory(10000)


# Helper Functions

# Return the default type map associated with the default Catan board
def get_default_type_map():
    DEFAULT_TYPE_MAP = [1, 3, 4, 0, 2, 3, 2, 0, 4, 5, 4, 1, 4, 1, 0, 3, 2, 0, 3]
    return DEFAULT_TYPE_MAP


# Agent Setup

if AGENT_SELECTED == "Adam":
    agent = Adam()
    MODEL_PATH = "adam.pth"
    # Check if the "model.pth" file exists
    if os.path.exists(MODEL_PATH):
        print("Ahhh! I'm back!")
        agent.load_model(MODEL_PATH)
else:
    agent = Randy()


# Start the game
game_instance.start_game()


# Game Loop

# Variable to keep our game loop running
running = True

learn_steps = 0

while running:

    # Get the legal actions from the game instance
    actions = game_instance.get_legal_actions()

    # Get all possible actions from the game instance
    all_actions = game_instance.get_all_possible_actions()

    # Get the game state
    game_state = game_instance.get_state()

    # Get the agent to pick an action
    action = agent.select_action(game_state, all_actions, actions)

    # What is the index of the action when set against the entire list of actions?
    action_index = all_actions.index(action)

    # Take a step in the game
    game_instance.step(action)

    # Get the new game state
    new_game_state = game_instance.get_state()

    # Get game over flag
    game_over = game_instance.get_game_over_flag()

    # Get the reward from the Agent
    reward = agent.reward(
        game_state,
        action,
        new_game_state,
        actions,
        all_actions,
        game_over,
    )

    # Print reward
    print("Reward: ", reward)

    # Create a memory tuple
    memory_tuple = (game_state, action_index, reward, new_game_state)

    # Add the memory tuple to the replay buffer
    replay_memory.add(memory_tuple)

    # Is the size of the replay memory more than 16?
    if replay_memory.get_buffer_size() > 100:
        if learn_steps < 6:
            # Increment the learn steps
            learn_steps += 1
        else:
            # Reset the learn steps
            learn_steps = 0
            # Call the learn() method on the agent
            agent.learn(replay_memory)

    # Game over?
    game_over_flag = game_instance.get_game_over_flag()
    pulled_to_csv = True

    if game_over_flag == True:
        if pulled_to_csv == False:
            replay_memory.save_buffer()
            print("Saved as JSON!")
        pulled_to_csv = True

        # Reset the game
        game_instance.reset()

        # Start the game
        game_instance.start_game()

        # Agent saving
        if AGENT_SELECTED == "Adam":
            # Save the model first
            agent.save_model(MODEL_PATH)
