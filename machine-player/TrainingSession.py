# This class encapsulates a training session for a machine player.
# Whether an Agent is learning in headless mode or using our GUI, complexities of the training process are handled by this class.

# Imports
import os
import csv
from datetime import datetime
from CatanGame import CatanGame
from Randy import Randy
from Adam import Adam
from Redmond import Redmond
from Davish import Davish
from Gonzales import Gonzales
from errors import AgentCompatibilityError
import random


# Define the TrainingSession class
class TrainingSession:
    # Define the constructor
    def __init__(
        self,
        agent="Randy",
        games=10000,
        epsilon_dec=0.0005,
        min_epsilon=0.1,
        agent_learning_interval=12,
        other_learning_interval=24,
        board_dims=[3, 4, 5, 4, 3],
        opponents=["Randy"],
        use_pixels=False,
        number_of_players=1,
    ):
        self.use_pixels = use_pixels
        self.AGENT_SELECTED = agent
        self.OPPONENTS_SELECTED = opponents
        self.EPISODES = games
        self.EPSILON_DECAY = epsilon_dec
        self.MIN_EPSILON = min_epsilon
        self.AGENT_LEARNING_INTERVAL = agent_learning_interval
        self.OTHER_LEARNING_INTERVAL = other_learning_interval
        self.BOARD_DIMS = board_dims

        self.GAME_INSTANCE = CatanGame()

        self.ALL_ACTIONS = self.GAME_INSTANCE.get_all_possible_actions()

        self.AGENT = None
        self.MODEL_PATH = None
        self.set_agent(self.AGENT_SELECTED)

        self.OPPONENTS = []
        self.set_opponents()
        self.NUMBER_OF_PLAYERS = number_of_players
        self.PLAYER_QUEUE = []
        self.set_player_queue()
        self.agent_index = 0

        self.player_turn_pointer = 0

        self.reset_game_data_dict()
        self.reset_game_session_data_dict()
        self.data_analysis_filename = None

        self.running = False
        self.agent_learning_steps = 0
        self.other_learning_steps = 0
        self.games_played = 0
        self.wins_recorded_this_session = 0

        self.pixel_data = None
        self.pixel_data_previous = None

    # Single step through the game (one action)
    def time_step(self):
        if self.running:

            # Only get the list of all possible actions once
            if self.games_played == 0:
                self.ALL_ACTIONS = self.GAME_INSTANCE.get_all_possible_actions()

            # Put the value of the player turn pointer into a variable for easier use
            player = self.player_turn_pointer

            # Get all the legal actions for the current player
            legal_actions = self.GAME_INSTANCE.get_legal_actions(player)

            # Get the game state for the current player
            game_state = self.GAME_INSTANCE.get_state(player)

            # Request the current player take an action
            # Pixel players are no longer compatible with this version of the training session
            chosen_action = self.PLAYER_QUEUE[player].select_action(
                game_state, self.ALL_ACTIONS, legal_actions
            )

            # Increment some counters depending on the action
            # This is for the main AGENT only; will be updated for other players later in future versions
            if player == self.agent_index:
                if chosen_action not in legal_actions:
                    self.game_data_dict["total_illegal_actions_attempted"] += 1
                else:
                    split_action = chosen_action.split("_")
                    if split_action[0] == "build" and split_action[1] == "road":
                        self.game_data_dict["total_roads_built"] += 1

            # Get the index of the chosen action
            action_index = self.ALL_ACTIONS.index(chosen_action)

            # Take a step in the game
            self.GAME_INSTANCE.step(chosen_action, player)

            # Increment the total number of steps taken
            if player == self.agent_index:
                self.game_data_dict["total_steps_taken"] += 1

            # Get the new game state
            new_game_state = self.GAME_INSTANCE.get_state()

            # Get the game over flag
            game_over = self.GAME_INSTANCE.get_game_over_flag()

            # Get the reward information so we can pass it to the agent
            reward_information = self.GAME_INSTANCE.reward_information_request(
                chosen_action, legal_actions
            )

            # Get the reward from the Agent
            reward = self.PLAYER_QUEUE[player].reward(reward_information)

            # Increment the total reward points earned if the player is the main AGENT
            if player == self.agent_index:
                self.game_data_dict["total_reward_points_earned"] += reward

            # Set the value of 'done' for the memory tuple
            done = 0
            if game_over:
                done = 1

            # Create a memory tuple
            # No longer supports pixel players in this version of the training session
            memory_tuple = (
                game_state,
                action_index,
                reward,
                new_game_state,
                done,
            )

            # Feed the memory tuple to the agent
            self.PLAYER_QUEUE[player].feed_memory(memory_tuple)

            # Is it time to optimise the agent?
            if player == self.agent_index:
                if self.agent_learning_steps < self.AGENT_LEARNING_INTERVAL:

                    # Increment counter if we are not ready to learn
                    self.agent_learning_steps += 1

                else:

                    # Call the appropriate learning function if we are ready to learn
                    self.agent_learning_steps = 0
                    loss = self.AGENT.learn()
                    if loss is not None:
                        self.game_data_dict["loss"].append(loss)

            # Is it time to optimise the other players?
            else:
                if self.other_learning_steps < self.OTHER_LEARNING_INTERVAL:

                    # Increment counter if we are not ready to learn
                    self.other_learning_steps += 1

                else:

                    # Call the appropriate learning function if we are ready to learn
                    self.other_learning_steps = 0
                    self.PLAYER_QUEUE[player].learn()

            # Complete some post-game tasks if the game is over
            if game_over:

                # Increment the number of games played
                self.games_played += 1

                # Get the current winner of the game
                winner = self.GAME_INSTANCE.get_player_id_of_current_winner()

                # If the winner is the agent, increment the wins recorded this session
                if winner == self.agent_index:
                    self.wins_recorded_this_session += 1

                # Calculate the average loss for the last game
                loss_list = self.game_data_dict["loss"]
                if len(loss_list) > 0:
                    average_loss = sum(loss_list) / len(loss_list)
                else:
                    average_loss = 0

                # Update the game session data dictionaries
                self.game_session_data_dict[
                    "total_illegal_actions_attempted_list"
                ].append(self.game_data_dict["total_illegal_actions_attempted"])
                self.game_session_data_dict["total_roads_built_list"].append(
                    self.game_data_dict["total_roads_built"]
                )
                self.game_session_data_dict["total_steps_taken_list"].append(
                    self.game_data_dict["total_steps_taken"]
                )
                self.game_session_data_dict["total_reward_points_earned_list"].append(
                    self.game_data_dict["total_reward_points_earned"]
                )
                self.game_session_data_dict["game_number_list"].append(
                    self.games_played
                )
                self.game_session_data_dict["turn_of_victory_list"].append(
                    self.GAME_INSTANCE.get_turn_number()
                )
                self.game_session_data_dict["epsilon_list"].append(
                    self.AGENT.get_exploration_rate()
                )
                self.game_session_data_dict["wins"].append(
                    (self.wins_recorded_this_session / self.games_played) * 100
                )
                self.game_session_data_dict["average_loss"].append(average_loss)

                # Output some game analysis to a .csv file
                if self.games_played == self.EPISODES or self.games_played % 100 == 0:
                    self.save_data_analysis_file()

                # Reduce the epsilon value of each player
                for player in self.PLAYER_QUEUE:

                    epsilon = player.get_exploration_rate()

                    if epsilon > self.MIN_EPSILON and player.get_nickname != "Randy":
                        epsilon -= self.EPSILON_DECAY
                        player.set_exploration_rate(epsilon)

                # Reset the game
                self.GAME_INSTANCE.reset(number_of_players=self.NUMBER_OF_PLAYERS)

                # Start the new game
                self.GAME_INSTANCE.start_game(number_of_players=self.NUMBER_OF_PLAYERS)
                self.ALL_ACTIONS = self.GAME_INSTANCE.get_all_possible_actions()

                # Reset the game data dictionary
                self.reset_game_data_dict()

                # Save the agent's state dict
                if (
                    self.AGENT_SELECTED == "Adam"
                    or self.AGENT_SELECTED == "Redmond"
                    or self.AGENT_SELECTED == "Davish"
                    or self.AGENT_SELECTED == "Gonzales"
                ):
                    self.AGENT.save_model(self.MODEL_PATH)

                # Save the state dict of the other players
                for player in self.PLAYER_QUEUE:
                    if (
                        player.get_nickname() != "Randy"
                        and player.get_nickname() != "Agent"
                        and player.get_nickname() != "Adam"
                    ):
                        file_name = player.get_nickname() + ".pth"
                        player.save_model(file_name)

                # Shuffle the turn order in preparation for the next game
                self.shuffle_turn_order()

            # Increment the player turn pointer if the action selected is to end the turn
            # If the player turn pointer is greater than the number of players, reset it to 0
            if chosen_action == "end_turn":
                self.player_turn_pointer += 1
                if self.player_turn_pointer >= self.NUMBER_OF_PLAYERS:
                    self.player_turn_pointer = 0

            # Return agent index (for GUI rendering)
            other_information = {
                "agent_index": self.agent_index,
            }

            # Return game state info
            return (
                self.running,
                legal_actions,
                chosen_action,
                self.games_played,
                self.player_turn_pointer,
                other_information,
            )

    # Method for starting the game loop
    def start(self, game_in_progress=False, players=1):
        # Variable to keep our game loop running
        self.running = True

        # Set the number of players
        self.NUMBER_OF_PLAYERS = players
        self.player_turn_pointer = 0

        # Shuffle the turn order straight away
        self.shuffle_turn_order()

        # Set the filename for the data analysis file
        self.set_data_analysis_filename()

        # Setup a game in progress if that is switched on
        if game_in_progress:
            self.GAME_INSTANCE.setup_game_in_progress()

        else:
            # Start the game instance
            self.GAME_INSTANCE.start_game(number_of_players=players)

        # Return "running"
        return self.running

    # Method for loading an agent into the training session
    def set_agent(self, agent):
        self.AGENT_SELECTED = agent
        agent_already_exists = False

        if self.AGENT_SELECTED == "Adam":
            self.AGENT = Adam(exploration_rate=1.0)
            self.MODEL_PATH = "adam.pth"
            # Check if the "adam.pth" file exists
            if os.path.exists(self.MODEL_PATH):
                print("Ahhh! I'm back.\nAdam reporting for duty!")
                self.AGENT.load_model(self.MODEL_PATH)
                agent_already_exists = True
        elif self.AGENT_SELECTED == "Redmond":
            self.AGENT = Redmond(exploration_rate=1.0)
            self.MODEL_PATH = "redmond.pth"
            # Check if the "redmond.pth" file exists
            if os.path.exists(self.MODEL_PATH):
                print("So, you've awoken me again...\nRedmond reporting for duty!")
                self.AGENT.load_model(self.MODEL_PATH)
                agent_already_exists = True
        elif self.AGENT_SELECTED == "Davish":
            self.AGENT = Davish(exploration_rate=1.0)
            self.MODEL_PATH = "davish.pth"
            # Check if the "davish.pth" file exists
            if os.path.exists(self.MODEL_PATH):
                print("Gold Five, standing by. Stay on target!")
                self.AGENT.load_model(self.MODEL_PATH)
                agent_already_exists = True
            else:
                print("Creating agent Davish for the first time.")
        elif self.AGENT_SELECTED == "Gonzales":
            self.AGENT = Gonzales(exploration_rate=1.0)
            self.MODEL_PATH = "gonzales.pth"
            # Check if the "gonzales.pth" file exists
            if os.path.exists(self.MODEL_PATH):
                print("Speedy Gonzales, reporting for duty!")
                self.AGENT.load_model(self.MODEL_PATH)
                agent_already_exists = True
            else:
                print("Creating agent Gonzales for the first time.")
        else:
            self.AGENT = Randy()

        if agent_already_exists:
            self.AGENT.set_exploration_rate(self.MIN_EPSILON)

        # Raise an error if an agent has been loaded that isn't compatible with input data
        if self.AGENT.get_pixel_compatible() != self.use_pixels:
            raise AgentCompatibilityError(
                "Agent selected not compatible with input data."
            )

    # Method for loading an opponent into the training session
    def set_opponents(self):
        # Loop through all of the opponents and load them in
        for opponent_name in self.OPPONENTS_SELECTED:
            if opponent_name == "Adam":
                new_opponent = Adam(exploration_rate=0.1)
                path = "adam.pth"
                # Check if the "adam.pth" file exists
                if os.path.exists(path):
                    print("I am Adam, and I am your opponent!")
                    # Load the model, and set it to evaluation mode
                    new_opponent.load_model(path)
                    new_opponent.model.eval()
            elif opponent_name == "Randy":
                new_opponent = Randy()

            # Agent isn't Randy or Adam, so will be an Adam agent with a nickname
            else:
                new_opponent = Adam(exploration_rate=1, override_nickname=opponent_name)
                path = opponent_name + ".pth"
                # Check if the ".pth" file exists
                if os.path.exists(path):
                    print("Loading in opponent " + opponent_name + " from file.")
                    # Load the model
                    new_opponent.load_model(path)
                    # Set the exploration rate to 0.1
                    new_opponent.set_exploration_rate(0.1)

            # Add the new opponent to the list of opponents
            self.OPPONENTS.append(new_opponent)

    # Method for setting the filename for the data analysis file
    def set_data_analysis_filename(self):
        # Information for file keeping
        now = datetime.now()
        self.data_analysis_filename = (
            "training_session_data/"
            + self.AGENT_SELECTED
            + now.strftime("-%Y-%m-%d-%H-%M")
            + ".csv"
        )

    # Method for saving the data analysis file
    def save_data_analysis_file(self):
        data = zip(
            self.game_session_data_dict["game_number_list"],
            self.game_session_data_dict["total_illegal_actions_attempted_list"],
            self.game_session_data_dict["total_steps_taken_list"],
            self.game_session_data_dict["total_reward_points_earned_list"],
            self.game_session_data_dict["total_roads_built_list"],
            self.game_session_data_dict["turn_of_victory_list"],
            self.game_session_data_dict["epsilon_list"],
            self.game_session_data_dict["wins"],
            self.game_session_data_dict["average_loss"],
        )

        # Check if file exists
        if not os.path.isfile(self.data_analysis_filename):
            # File does not exist, open for writing and write header row
            with open(self.data_analysis_filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "Game number",
                        "Total illegal actions attempted",
                        "Total steps taken",
                        "Total reward points",
                        "Roads built",
                        "Turn where victory achieved",
                        "Epsilon",
                        "Win percentage",
                        "Average loss",
                    ]
                )

        # Open the file for writing and write the data rows
        with open(self.data_analysis_filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for row in data:
                writer.writerow(row)

        # Reset the game session data dictionary
        self.reset_game_session_data_dict()

    # Return the default type map associated with the default Catan board
    def get_default_type_map():
        default_type_map = [1, 3, 4, 0, 2, 3, 2, 0, 4, 5, 4, 1, 4, 1, 0, 3, 2, 0, 3]
        return default_type_map

    # Reset the game data dictionary
    def reset_game_data_dict(self):
        self.game_data_dict = {
            "total_illegal_actions_attempted": 0,
            "total_steps_taken": 0,
            "total_reward_points_earned": 0,
            "total_roads_built": 0,
            "loss": [],
        }

    # Reset the game session data dictionary
    def reset_game_session_data_dict(self):
        self.game_session_data_dict = {
            "game_number_list": [],
            "total_illegal_actions_attempted_list": [],
            "total_steps_taken_list": [],
            "total_reward_points_earned_list": [],
            "total_roads_built_list": [],
            "turn_of_victory_list": [],
            "epsilon_list": [],
            "wins": [],
            "average_loss": [],
        }

    # Method for getting the game state
    def get_game_state(self):
        return self.GAME_INSTANCE.get_state()

    # Method for getting the board dims
    def get_board_dims(self):
        return self.BOARD_DIMS

    # Method for getting the game instance
    def get_game_instance(self):
        return self.GAME_INSTANCE

    # Method for feeding pixel data
    def feed_pixel_data(self, pixel_data):
        self.pixel_data_previous = self.pixel_data
        self.pixel_data = pixel_data

    # Method for shuffling the turn order
    def shuffle_turn_order(self):
        # Shuffle the PLAYER_QUEUE
        random.shuffle(self.PLAYER_QUEUE)
        # Get the index of the agent
        self.agent_index = self.PLAYER_QUEUE.index(self.AGENT)
        # Print the index of the agent
        # Can be disabled if desired
        # print("Agent index: " + str(self.agent_index))

    # Method for setting the player queue
    def set_player_queue(self):
        # Set the player queue
        self.PLAYER_QUEUE = [self.AGENT] + self.OPPONENTS
