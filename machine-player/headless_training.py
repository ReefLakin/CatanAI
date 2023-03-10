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

# DATETIME
from datetime import datetime

# CSV
import csv


# Set Constants

# BOARD DIMENSIONS
BOARD_DIMS = [3, 4, 5, 4, 3]

# THE ACTUAL BOARD
game_instance = CatanGame()

# THE AGENT
AGENT_SELECTED = "Adam"

# NUMBER OF GAMES (Set to -1 for infinite)
EPISODES = 1200

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
games_played = 0

# Other data about the current session
total_illegal_actions_attempted = 0
total_steps_taken = 0
total_reward_points_earned = 0
total_roads_built = 0

# Other data about the current session, but these are arrays
total_illegal_actions_attempted_list = []
total_steps_taken_list = []
total_reward_points_earned_list = []
total_roads_built_list = []
game_numbers_list = []

while running is True and games_played != EPISODES:

    # Get the legal actions from the game instance
    actions = game_instance.get_legal_actions()

    # Get all possible actions from the game instance
    all_actions = game_instance.get_all_possible_actions()

    # Get the game state
    game_state = game_instance.get_state()

    # Get the agent to pick an action
    action = agent.select_action(game_state, all_actions, actions)

    # If the action is illegal, increment the illegal action counter
    if action not in actions:
        total_illegal_actions_attempted += 1
    else:
        # If the action is to build a road, increment the road counter
        split_action = action.split("_")
        if split_action[0] == "build" and split_action[1] == "road":
            total_roads_built += 1

    # What is the index of the action when set against the entire list of actions?
    action_index = all_actions.index(action)

    # Take a step in the game
    game_instance.step(action)

    # Increment the total number of steps taken
    total_steps_taken += 1

    # Get the new game state
    new_game_state = game_instance.get_state()

    # Get game over flag
    game_over = game_instance.get_game_over_flag()

    # Get reward information from the game instance
    reward_information = game_instance.reward_information_request(action, actions)

    # Get the reward from the Agent
    reward = agent.reward(reward_information)

    # Increment the total reward points earned
    total_reward_points_earned += reward

    # Create a memory tuple
    memory_tuple = (game_state, action_index, reward, new_game_state)

    # Add the memory tuple to the replay buffer
    replay_memory.add(memory_tuple)

    # Is the size of the replay memory more than 16?
    if replay_memory.get_buffer_size() > 32:
        if learn_steps < 4:
            # Increment the learn steps
            learn_steps += 1
        else:
            # Reset the learn steps
            learn_steps = 0
            # Call the learn() method on the agent
            agent.learn(replay_memory)

    # Game over?
    game_over_flag = game_instance.get_game_over_flag()

    # Print the number of games played
    print("Games played: " + str(games_played))

    if game_over_flag == True:
        # Append the CSV data to the arrays
        total_illegal_actions_attempted_list.append(total_illegal_actions_attempted)
        total_steps_taken_list.append(total_steps_taken)
        total_reward_points_earned_list.append(total_reward_points_earned)
        total_roads_built_list.append(total_roads_built)
        game_numbers_list.append(games_played)

        games_played += 1  # Increment the number of games played
        if games_played == EPISODES or games_played == 12000:
            if games_played == 12000:
                games_played = 0
            replay_memory.save_buffer(AGENT_SELECTED)

            data = zip(
                game_numbers_list,
                total_illegal_actions_attempted_list,
                total_steps_taken_list,
                total_reward_points_earned_list,
                total_roads_built_list,
            )

            # Get current date and time for file name
            now = datetime.now()
            filename = (
                "training_session_data/"
                + AGENT_SELECTED
                + now.strftime("-%Y-%m-%d-%H-%M")
                + ".csv"
            )

            # Open a file for writing
            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Write the header row containing the array names
                writer.writerow(
                    [
                        "Game number",
                        "Total illegal actions attempted",
                        "Total steps taken",
                        "Total reward points",
                        "Roads built",
                    ]
                )

                # Write the data rows
                for row in data:
                    writer.writerow(row)

        # Reset the game
        game_instance.reset()

        # Reset other data about the current session
        total_illegal_actions_attempted = 0
        total_steps_taken = 0
        total_reward_points_earned = 0
        total_roads_built = 0

        # Start the game
        game_instance.start_game()

        # Agent saving
        if AGENT_SELECTED == "Adam":
            # Save the model first
            agent.save_model(MODEL_PATH)
