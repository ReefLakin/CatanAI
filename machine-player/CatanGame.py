# Imports
from Board import Board
from Tile import Tile
import random


class CatanGame:
    def __init__(self):
        # Initialize the game board and other necessary variables here
        self.reset()

    def reset(self):
        # Reset the game board to its starting state
        # For now, we will use the default board
        # Later, we will add the ability to specify a custom board
        board_dims = [3, 4, 5, 4, 3]
        tile_values = [10, 2, 9, 12, 6, 4, 10, 9, 11, 0, 3, 8, 8, 3, 4, 5, 5, 6, 11]
        tile_types = [
            "ore",
            "wool",
            "lumber",
            "grain",
            "brick",
            "wool",
            "brick",
            "grain",
            "lumber",
            "desert",
            "lumber",
            "ore",
            "lumber",
            "ore",
            "grain",
            "wool",
            "brick",
            "grain",
            "wool",
        ]
        # Create the tiles
        tiles = []
        for i in range(19):
            temp_tile = Tile(
                tile_types[i],
                i,
                False,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                tile_values[i],
                None,
                None,
                None,
            )
            tiles.append(temp_tile)
        # Create the board
        self.board = Board(board_dims, tiles)
        # Set up the resource pool
        self.resource_pool = {"lumber": 0, "brick": 0, "wool": 0, "grain": 0, "ore": 0}
        # Set up the player's VP count
        self.victory_points = 0
        # Set up the turn number
        self.turn_number = 1
        # Set up the reward attribute
        self.reward = 0
        # Set the game over flag
        self.game_over = False
        # Set all possible actions
        self.set_all_possible_actions()
        # Set the legal actions
        self.set_legal_actions()
        # Set the dice roll
        self.most_recent_roll = (0, 0, 0, "Dice haven't been rolled yet.")

    def get_board(self):
        # Return the game board
        return self.board

    def step(self, action):
        # Print the action taken to the console along with turn number
        print("Turn " + str(self.turn_number) + ": " + action)

        # Take a step in the game by applying the given action
        action_parts = action.split("_")

        # Is the action even legal?
        if action not in self.legal_actions:
            reward = self.get_reward(action)
            self.reward = self.reward + reward

        # If the action is legal, is the action a 4:1 trade?
        elif action_parts[0] == "trade":
            # Reduce the traded resource by 4
            self.resource_pool[action_parts[5]] = (
                self.resource_pool[action_parts[5]] - 4
            )
            # Increase the received resource by 1
            self.resource_pool[action_parts[6]] = (
                self.resource_pool[action_parts[6]] + 1
            )

        # If the action is legal, is it a road build?
        elif action_parts[0] == "build" and action_parts[1] == "road":
            # Call a currently non-existent function to build a road
            self.board.build_road(
                int(action_parts[3]),
                int(action_parts[4]),
                int(action_parts[5]),
                action_parts[2],
            )

            # Remove 1 lumber and 1 brick from the resource pool
            self.resource_pool["lumber"] = self.resource_pool["lumber"] - 1
            self.resource_pool["brick"] = self.resource_pool["brick"] - 1

        # If the action is legal, is it a settlement build?
        elif action_parts[0] == "build" and action_parts[1] == "settlement":
            # Call a currently non-existent function to build a settlement
            self.board.build_settlement(
                int(action_parts[3]),
                int(action_parts[4]),
                int(action_parts[5]),
                action_parts[2],
            )

            # Update the player's VP count
            self.victory_points = self.victory_points + 1

            # Remove 1 lumber, 1 brick, 1 wool, and 1 grain from the resource pool
            self.resource_pool["lumber"] = self.resource_pool["lumber"] - 1
            self.resource_pool["brick"] = self.resource_pool["brick"] - 1
            self.resource_pool["wool"] = self.resource_pool["wool"] - 1
            self.resource_pool["grain"] = self.resource_pool["grain"] - 1

        # If the action is legal, is it a simple end turn?
        elif action == "end_turn":

            # Update the turn number
            self.turn_number = self.turn_number + 1

            # Roll the dice
            roll = self.dice_roll()

            # Distribute resources
            self.distribute_resources(roll)

        # Update the reward
        reward = self.get_reward(action)
        self.reward = self.reward + reward

        # Is the game over?
        if self.victory_points >= 10:
            self.game_over = True

        # Update the legal actions
        self.set_legal_actions()

    def get_reward(self, action):
        # Return the reward for taking a given action
        # Picking an illegal action will result in a reward of -1
        # If the player has earned 10 VPs, they have won the game and will receive a reward of +100
        if self.victory_points >= 10:
            return 500
        if action not in self.legal_actions:
            return -1
        else:
            split_action = action.split("_")
            # Building a road will net a reward of +10
            if split_action[0] == "build" and split_action[1] == "road":
                return 5
            # Building a settlement will net a reward of +30
            elif split_action[0] == "build" and split_action[1] == "settlement":
                return 100
            # Anything else will net a reward of 0
            else:
                return 0

    def get_current_game_reward(self):
        # Return the current game reward
        return self.reward

    def select_action(self, action_values, epsilon):
        # Select an action to take based on the given action values and the given probability of taking a random action (epsilon)
        # There are many different approaches to selecting actions, such as greedily choosing the action with the highest predicted value or using a probability distribution over the predicted values to sample an action
        pass

    def get_state(self):
        # Build a dictionary of state information, which includes all of the tile state info, player resources, VPs and turn number.
        # This info is purposely in a human-readable state, but will likely be preprocessed before being fed to the DQN.
        return {
            "side_states": self.board.get_side_states(),  # 72
            "vertex_states": self.board.get_vertex_states(),  # 54
            "board_dims": self.board.get_board_dims(),  # 5
            "tile_types": self.board.get_tile_types_in_a_list(),  # 19
            "num_brick": self.resource_pool["brick"],  # 1
            "num_lumber": self.resource_pool["lumber"],  # 1
            "num_wool": self.resource_pool["wool"],  # 1
            "num_grain": self.resource_pool["grain"],  # 1
            "num_ore": self.resource_pool["ore"],  # 1
            "victory_points": self.victory_points,  # 1
            "turn_number": self.turn_number,  # 1
            "tile_values": self.board.get_tile_numbers_in_a_list(),  # 19
            # Total: 176
        }

    def get_state_as_single_list(self):
        # Build a list of state information
        # This is put in one array and can be used for replay memory or preprocessing, etc.
        listed_state = []
        current_state = self.get_state()

        # Victory points
        listed_state.append(current_state["victory_points"])

        # Turn number
        listed_state.append(current_state["turn_number"])

        # Rows in board
        listed_state.append(len(current_state["board_dims"]))

        # Total number of tiles
        listed_state.append(len(current_state["tile_types"]))

    def set_legal_actions(self):
        # Return a list of legal actions given the current state of the game
        self.legal_actions = []
        # Loop over each action from the set of all possible actions
        for action in self.all_actions:
            # Split the action into its parts
            action_parts = action.split("_")
            # If the player does not have enough resources to build a settlement, skip this action; consider it cut from the list of legal actions
            if (action_parts[0] == "build" and action_parts[1] == "settlement") and (
                self.resource_pool["brick"] < 1
                or self.resource_pool["wool"] < 1
                or self.resource_pool["lumber"] < 1
                or self.resource_pool["grain"] < 1
            ):
                continue
            # If the player does not have enough resources to build a road, skip this action; consider it cut from the list of legal actions
            elif (action_parts[0] == "build" and action_parts[1] == "road") and (
                self.resource_pool["brick"] < 1 or self.resource_pool["lumber"] < 1
            ):
                continue
            # If the location already has a settlement on it, skip this action; consider it cut from the list of legal actions
            elif action_parts[0] == "build" and action_parts[1] == "settlement":
                direction = action_parts[2]
                q_coord = int(action_parts[3])
                r_coord = int(action_parts[4])
                s_coord = int(action_parts[5])
                tile = self.board.get_tile(q_coord, r_coord, s_coord)
                # Raise an error if the tile is not found; that shouldn't happen here
                if tile is None:
                    raise ValueError(
                        "Tile not found at coordinates ({}, {}, {})".format(
                            q_coord, r_coord, s_coord
                        )
                    )
                vert_val = tile.get_vertex_from_direction(direction)
                if vert_val is not None:
                    continue
                else:
                    # Assuming a settlement is not already on the board in that position, check if the settlement placement is legal
                    settlement_legal = self.board.check_settlement_placement_legal(
                        q_coord, r_coord, s_coord, direction
                    )
                    if settlement_legal == False:
                        continue
                    else:
                        self.legal_actions.append(action)
            # If the location already has a road on it, skip this action; consider it cut from the list of legal actions
            elif action_parts[0] == "build" and action_parts[1] == "road":
                direction = action_parts[2]
                q_coord = int(action_parts[3])
                r_coord = int(action_parts[4])
                s_coord = int(action_parts[5])
                tile = self.board.get_tile(q_coord, r_coord, s_coord)
                # Raise an error if the tile is not found; that shouldn't happen here
                if tile is None:
                    raise ValueError(
                        "Tile not found at coordinates ({}, {}, {})".format(
                            q_coord, r_coord, s_coord
                        )
                    )
                side_val = tile.get_side_from_direction(direction)
                if side_val is not None:
                    continue
                else:
                    # Assuming a road is not already on the board in that position, check if the road placement is legal
                    road_legal = self.board.check_road_placement_legal(
                        q_coord, r_coord, s_coord, direction
                    )
                    if road_legal == False:
                        continue
                    else:
                        self.legal_actions.append(action)
            # If the player tries to make a 4:1 trade with the bank, check if they have enough resources to make the trade
            elif (
                action_parts[0] == "trade"
                and action_parts[1] == "bank"
                and action_parts[2] == "4"
                and action_parts[3] == "for"
                and action_parts[4] == "1"
            ):
                if self.resource_pool[action_parts[5]] < 4:
                    continue
                else:
                    self.legal_actions.append(action)
            # At this stage, the action is legal, so add it to the list of legal actions
            else:
                self.legal_actions.append(action)

    def get_legal_actions(self):
        # Return the list of legal actions
        return self.legal_actions

    def set_all_possible_actions(self):
        # Set the list of all possible actions, usually run once at the beginning of the game
        self.all_actions = [
            "end_turn",
            "trade_bank_4_for_1_lumber_brick",
            "trade_bank_4_for_1_lumber_wool",
            "trade_bank_4_for_1_lumber_grain",
            "trade_bank_4_for_1_lumber_ore",
            "trade_bank_4_for_1_brick_wool",
            "trade_bank_4_for_1_brick_grain",
            "trade_bank_4_for_1_brick_ore",
            "trade_bank_4_for_1_brick_lumber",
            "trade_bank_4_for_1_wool_grain",
            "trade_bank_4_for_1_wool_ore",
            "trade_bank_4_for_1_wool_lumber",
            "trade_bank_4_for_1_wool_brick",
            "trade_bank_4_for_1_grain_ore",
            "trade_bank_4_for_1_grain_lumber",
            "trade_bank_4_for_1_grain_brick",
            "trade_bank_4_for_1_grain_wool",
            "trade_bank_4_for_1_ore_lumber",
            "trade_bank_4_for_1_ore_brick",
            "trade_bank_4_for_1_ore_wool",
            "trade_bank_4_for_1_ore_grain",
        ]

        # Add all possible settlement locations to the list of actions
        for tile in self.board.get_board_tiles():
            q_coord = tile.get_q_coord()
            r_coord = tile.get_r_coord()
            s_coord = tile.get_s_coord()
            self.all_actions.append(
                f"build_settlement_northwest_{q_coord}_{r_coord}_{s_coord}"
            )
            self.all_actions.append(
                f"build_settlement_north_{q_coord}_{r_coord}_{s_coord}"
            )
            self.all_actions.append(
                f"build_settlement_northeast_{q_coord}_{r_coord}_{s_coord}"
            )
            self.all_actions.append(
                f"build_settlement_southeast_{q_coord}_{r_coord}_{s_coord}"
            )
            self.all_actions.append(
                f"build_settlement_south_{q_coord}_{r_coord}_{s_coord}"
            )
            self.all_actions.append(
                f"build_settlement_southwest_{q_coord}_{r_coord}_{s_coord}"
            )
            self.all_actions.append(
                f"build_road_northwest_{q_coord}_{r_coord}_{s_coord}"
            )
            self.all_actions.append(
                f"build_road_northeast_{q_coord}_{r_coord}_{s_coord}"
            )
            self.all_actions.append(f"build_road_east_{q_coord}_{r_coord}_{s_coord}")
            self.all_actions.append(
                f"build_road_southeast_{q_coord}_{r_coord}_{s_coord}"
            )
            self.all_actions.append(
                f"build_road_southwest_{q_coord}_{r_coord}_{s_coord}"
            )
            self.all_actions.append(f"build_road_west_{q_coord}_{r_coord}_{s_coord}")

        # Print the size of the list of all possible actions (246)
        # print(f"Size of all possible actions: {len(self.all_actions)}")

    def distribute_resources(self, roll):
        # Distribute resources to players based on the given dice roll
        if roll != 7:
            tiles = self.board.get_board_tiles()
            for tile in tiles:
                if tile.get_tile_value() == roll:
                    occupied = tile.get_occupied_verticies()
                    resource_to_give = tile.get_type()
                    resource_count = len(occupied)

                    # Give the correct resource to the player
                    self.resource_pool[resource_to_give] += resource_count

    def get_all_possible_actions(self):
        # Return the list of all possible actions
        return self.all_actions

    def dice_roll(self):
        # Roll the dice and return the result
        dice_1 = random.randint(1, 6)
        dice_2 = random.randint(1, 6)
        # Print the roll to the console
        print(f"Rolled a {dice_1} and a {dice_2} for a total of {dice_1 + dice_2}")
        # Store the roll in the game state
        self.most_recent_roll = (
            dice_1,
            dice_2,
            dice_1 + dice_2,
            f"{dice_1} and {dice_2}",
        )
        return dice_1 + dice_2

    def get_most_recent_roll(self):
        # Return the most recent dice roll
        return self.most_recent_roll

    def set_legal_actions_manually(self, actions):
        # Set the list of legal actions
        self.legal_actions = actions

    def setup_game_in_progress(self):
        # For testing purposes, set the game up as if it is in progress

        # Build a settlement at [0, -1, +1] southwest
        self.board.build_settlement(0, -1, 1, "southwest")
        # Build a settlement at [-1, 1, 0] northwest
        self.board.build_settlement(-1, 1, 0, "northwest")
        # Build a settlement at [0, -2, +2] southwest
        self.board.build_settlement(0, -2, 2, "southwest")
        # Build a road at [0, -1, +1] west
        self.board.build_road(0, -1, 1, "west")
        # Build a road at [0, -2, +2] southwest
        self.board.build_road(0, -2, 2, "southwest")
        # Build a road at [0, -1, +1] southwest
        self.board.build_road(0, -1, 1, "southwest")
        # Build a road at [0, 0, 0] west
        self.board.build_road(0, 0, 0, "west")
        # Build a settlement at [-1, 1, 0] northwest
        self.board.build_road(-1, 1, 0, "northwest")
        # Set victory points to 5
        self.victory_points = 5
        # Give the player 3 brick, 1 ore and 2 lumber
        self.resource_pool["brick"] = 3
        self.resource_pool["ore"] = 1
        self.resource_pool["lumber"] = 2

    def get_game_over_flag(self):
        # Return the game over flag
        return self.game_over

    def start_game(self):
        # Set a flag so that the game knows the pre-game build phase has begun
        self.build_phase_active = True

        # For now, we're going to build for the player to start the game

        # Build a settlement at [1, -1, 0] northeast
        self.board.build_settlement(1, -1, 0, "northeast")
        # Build a road at [1, -1, 0] east
        self.board.build_road(1, -1, 0, "east")
        # Build a settlement at [-1, 1, 0] south
        self.board.build_settlement(-1, 1, 0, "south")
        # Build a road at [-1, 1, 0] southeast
        self.board.build_road(-1, 1, 0, "southeast")

        # Set the victory points to 2
        self.victory_points = 2

        # Turn off the build phase flag
        self.build_phase_active = False

        # Set legal actions
        self.set_legal_actions()
