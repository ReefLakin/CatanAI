# This class encapsulates a training session for a machine player.
# Whether an Agent is learning in headless mode or using our GUI, complexities of the training process are handled by this class.

# Imports
import os
import csv
from datetime import datetime
from CatanGame import CatanGame
from Randy import Randy
from Adam import Adam
from Phil import Phil
from Redmond import Redmond
from Eugene import Eugene

# Define the TrainingSession class
class TrainingSession:
    # Define the constructor
    def __init__(
        self,
        agent="Randy",
        games=10000,
        epsilon_dec=0.0003,
        min_epsilon=0.1,
        learning_interval=3,
        board_dims=[3, 4, 5, 4, 3],
    ):

        self.AGENT_SELECTED = agent
        self.EPISODES = games
        self.EPSILON_DECAY = epsilon_dec
        self.MIN_EPSILON = min_epsilon
        self.LEARNING_INTERVAL = learning_interval
        self.BOARD_DIMS = board_dims

        self.GAME_INSTANCE = CatanGame()

        self.AGENT = None
        self.MODEL_PATH = None
        self.set_agent(self.AGENT_SELECTED)

        self.OPPONENT = Randy()
        self.NUMBER_OF_PLAYERS = 1
        self.PLAYER_QUEUE = [self.AGENT, self.OPPONENT]

        self.player_turn_pointer = 0

        self.reset_game_data_dict()
        self.reset_game_session_data_dict()
        self.data_analysis_filename = None

        self.running = False
        self.learning_steps = 0
        self.games_played = 0
        self.wins_recorded_this_session = 0

    # Method for continuing the game loop
    def time_step(self):
        if self.running:
            # Get a list of all possible actions from the game instance, even if they are illegal
            all_actions = self.GAME_INSTANCE.get_all_possible_actions()
            # Get a list of all legal actions from the game instance
            legal_actions = self.GAME_INSTANCE.get_legal_actions(
                player_id=self.player_turn_pointer
            )
            # Get the current state of the game instance
            game_state = self.GAME_INSTANCE.get_state(self.player_turn_pointer)

            # Agent selects an action
            chosen_action = self.PLAYER_QUEUE[self.player_turn_pointer].select_action(
                game_state, all_actions, legal_actions
            )

            # If the action is illegal, increment the illegal action counter, else check if we need to increment the road counter
            # This is only if the AGENT is taking a turn, not any of the opponents
            if self.player_turn_pointer == 0:
                if chosen_action not in legal_actions:
                    self.game_data_dict["total_illegal_actions_attempted"] += 1
                else:
                    split_action = chosen_action.split("_")
                    if split_action[0] == "build" and split_action[1] == "road":
                        self.game_data_dict["total_roads_built"] += 1

            # What is the index of the action when set against the entire list of actions?
            action_index = all_actions.index(chosen_action)

            # Take a step in the game
            self.GAME_INSTANCE.step(chosen_action, self.player_turn_pointer)

            # Increment the total number of steps taken
            # This is only if the AGENT is taking a turn, not any of the opponents
            if self.player_turn_pointer == 0:
                self.game_data_dict["total_steps_taken"] += 1

            # Get the new game state
            new_game_state = self.GAME_INSTANCE.get_state()

            # Get the game over flag
            game_over = self.GAME_INSTANCE.get_game_over_flag()

            if self.player_turn_pointer == 0:

                # Get reward information from the game instance
                reward_information = self.GAME_INSTANCE.reward_information_request(
                    chosen_action, legal_actions
                )

                # Get the reward from the Agent
                reward = self.AGENT.reward(reward_information)

                # Increment the total reward points earned
                self.game_data_dict["total_reward_points_earned"] += reward

                # Set the value of 'done' for the memory tuple
                done = 0
                if game_over:
                    done = 1

                # Create a memory tuple
                memory_tuple = (game_state, action_index, reward, new_game_state, done)

                # Feed the memory tuple to the agent
                if self.AGENT_SELECTED == "Phil":
                    # Find the index of the action in the list of all actions
                    action_with_idx = all_actions.index(chosen_action)
                    self.AGENT.store_transition(
                        game_state, action_with_idx, reward, new_game_state, game_over
                    )
                else:
                    self.AGENT.feed_memory(memory_tuple)

                # Increment the steps until call learning
                self.steps_until_call_learning += 1

                if self.learning_steps < self.LEARNING_INTERVAL:
                    # Increment the learn steps if we are not ready to learn
                    self.learning_steps += 1
                else:
                    # Reset the learn steps if we are ready to learn
                    self.learning_steps = 0
                    # Call the learn() method on the agent
                    loss = self.AGENT.learn()
                    # Add the loss to the game data dictionary only if it is not None
                    if loss is not None:
                        self.game_data_dict["loss"].append(loss)

            # Complete some post-game tasks if the game is over
            if game_over:
                # Increment the number of games played
                self.games_played += 1

                # Get the current winner of the game
                winner = self.GAME_INSTANCE.get_player_id_of_current_winner()

                # If the winner is the agent, increment the wins recorded this session
                if winner == 0:
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

                # Get the agents current epsilon value
                epsilon = self.AGENT.get_exploration_rate()

                # If the epsilon value is greater than the minimum epsilon value, reduce it
                if epsilon > self.MIN_EPSILON and self.AGENT_SELECTED != "Randy":
                    epsilon -= self.EPSILON_DECAY
                    self.AGENT.set_exploration_rate(epsilon)

                # Reset the game
                self.GAME_INSTANCE.reset(number_of_players=self.NUMBER_OF_PLAYERS)

                # Start the new game
                self.GAME_INSTANCE.start_game(number_of_players=self.NUMBER_OF_PLAYERS)

                # Reset the game data dictionary
                self.reset_game_data_dict()

                # Save the model's state dict
                if (
                    self.AGENT_SELECTED == "Adam"
                    or self.AGENT_SELECTED == "Redmond"
                    or self.AGENT_SELECTED == "Eugene"
                ):
                    self.AGENT.save_model(self.MODEL_PATH)

            # Increment the player turn pointer if the action selected is to end the turn
            # If the player turn pointer is greater than the number of players, reset it to 0
            if chosen_action == "end_turn":
                self.player_turn_pointer += 1
                if self.player_turn_pointer >= self.NUMBER_OF_PLAYERS:
                    self.player_turn_pointer = 0

            return self.running, legal_actions, chosen_action, self.games_played

    # Method for starting the game loop
    def start(self, game_in_progress=False, players=1):
        # Variable to keep our game loop running
        self.running = True
        self.steps_until_call_learning = 0

        # Set the number of players
        self.NUMBER_OF_PLAYERS = players
        self.player_turn_pointer = 0

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
        elif self.AGENT_SELECTED == "Phil":
            self.AGENT = Phil(
                gamma=0.9,
                epsilon=1.0,
                lr=0.001,
                input_dims=(525,),
                batch_size=64,
                n_action=382,
            )
        else:
            self.AGENT = Randy()

        if agent_already_exists:
            self.AGENT.set_exploration_rate(self.MIN_EPSILON)

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
