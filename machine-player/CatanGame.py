# Imports
from Board import Board
from Tile import Tile
import random


class CatanGame:
    def __init__(self):
        # Initialize the game board and other necessary variables here
        self.reset()

    def reset(self, number_of_players=1):
        # Reset the game board to its starting state
        # For now, we will use the default board
        # Later, we will add the ability to specify a custom board
        board_dims = [3, 4, 5, 4, 3]
        self.number_of_players = number_of_players
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
        self.resource_pool = [
            {"lumber": 0, "brick": 0, "wool": 0, "grain": 0, "ore": 0}
            for i in range(number_of_players)
        ]
        # Set up the player's VP count
        self.victory_points = 0
        # Set up the turn number
        self.turn_number = 1
        # Set up the reward attribute
        self.reward = 0
        # Set the game over flag
        self.game_over = False
        # Set the robbery_in_progress flag
        self.robbery_in_progress = False
        # Set all possible actions
        self.set_all_possible_actions()
        # Set the legal actions
        self.set_legal_actions()
        # Set the dice roll
        self.most_recent_roll = (0, 0, 0, "Dice haven't been rolled yet.")
        # Set the number of resources generated with the last roll
        self.most_recent_resources_generated = 0

    def get_board(self):
        # Return the game board
        return self.board

    def step(self, action, player_number=1):
        # Print the action taken to the console along with turn number
        print("Turn " + str(self.turn_number) + ": " + action)

        # Take a step in the game by applying the given action
        action_parts = action.split("_")

        # Is the action even legal?
        if action not in self.legal_actions[player_number - 1]:
            pass

        # If the action is legal, is the action a 4:1 trade?
        elif action_parts[0] == "trade":
            # Reduce the traded resource by 4
            self.resource_pool[player_number - 1][action_parts[5]] = (
                self.resource_pool[player_number - 1][action_parts[5]] - 4
            )
            # Increase the received resource by 1
            self.resource_pool[player_number - 1][action_parts[6]] = (
                self.resource_pool[player_number - 1][action_parts[6]] + 1
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
            self.resource_pool[player_number - 1]["lumber"] = (
                self.resource_pool[player_number - 1]["lumber"] - 1
            )
            self.resource_pool[player_number - 1]["brick"] = (
                self.resource_pool[player_number - 1]["brick"] - 1
            )

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
            self.resource_pool[player_number - 1]["lumber"] = (
                self.resource_pool[player_number - 1]["lumber"] - 1
            )
            self.resource_pool[player_number - 1]["brick"] = (
                self.resource_pool[player_number - 1]["brick"] - 1
            )
            self.resource_pool[player_number - 1]["wool"] = (
                self.resource_pool[player_number - 1]["wool"] - 1
            )
            self.resource_pool[player_number - 1]["grain"] = (
                self.resource_pool[player_number - 1]["grain"] - 1
            )

        # If the action is legal, is it a city build?
        elif action_parts[0] == "build" and action_parts[1] == "city":
            # Call a currently non-existent function to build a city
            self.board.build_city(
                int(action_parts[3]),
                int(action_parts[4]),
                int(action_parts[5]),
                action_parts[2],
            )

            # Update the player's VP count
            self.victory_points = self.victory_points + 1

            # Remove 2 grain and 3 ore from the resource pool
            self.resource_pool[player_number - 1]["grain"] = (
                self.resource_pool[player_number - 1]["grain"] - 2
            )
            self.resource_pool[player_number - 1]["ore"] = (
                self.resource_pool[player_number - 1]["ore"] - 3
            )

        # If the action is legal, is it a simple end turn?
        elif action == "end_turn":

            # Update the turn number
            self.turn_number = self.turn_number + 1

            # Roll the dice
            roll = self.dice_roll()

            # Distribute resources
            self.distribute_resources(roll)

        # If the action is legal, is it a robber move?
        elif action_parts[0] == "move" and action_parts[1] == "robber":
            # Move the robber
            self.board.move_robber(
                int(action_parts[2]), int(action_parts[3]), int(action_parts[4])
            )

            # Set the robbery_in_progress flag
            self.robbery_in_progress = False

        # Is the game over?
        if self.victory_points >= 10:
            self.game_over = True

        # Update the legal actions
        self.set_legal_actions()

    def get_current_game_reward(self):
        # Return the current game reward
        return self.reward

    def get_state(self, player_number=1):
        # Build a dictionary of state information, which includes all of the tile state info, player resources, VPs and turn number.
        # This info is purposely in a human-readable state, but will likely be preprocessed before being fed to the DQN.
        return {
            "side_states": self.board.get_side_states(),
            "side_owners": self.board.get_side_owners(),
            "vertex_states": self.board.get_vertex_states(),
            "vertex_owners": self.board.get_vertex_owners(),
            "board_dims": self.board.get_board_dims(),
            "tile_types": self.board.get_tile_types_in_a_list(),
            "num_brick": self.resource_pool[player_number - 1]["brick"],
            "num_lumber": self.resource_pool[player_number - 1]["lumber"],
            "num_wool": self.resource_pool[player_number - 1]["wool"],
            "num_grain": self.resource_pool[player_number - 1]["grain"],
            "num_ore": self.resource_pool[player_number - 1]["ore"],
            "victory_points": self.victory_points,
            "tile_values": self.board.get_tile_numbers_in_a_list(),
            "robber_states": self.board.get_robber_states(),
            "most_recent_roll": self.most_recent_roll[2],
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
        # Set legal actions for each player in the game
        self.legal_actions = [[] for i in range(self.number_of_players)]
        # PLEASE NOTE: player_number here is 0-indexed
        for player_number in range(self.number_of_players):
            # Loop over each action from the set of all possible actions
            for action in self.all_actions:
                # Split the action into its parts
                action_parts = action.split("_")

                if self.robbery_in_progress == True:
                    # Get the tile coordinates where the robber is currently located
                    robber_tile = self.board.get_robber_tile()
                    robber_tile_q = robber_tile.get_q_coord()
                    robber_tile_r = robber_tile.get_r_coord()
                    robber_tile_s = robber_tile.get_s_coord()

                    # If the robber is in progress, the only legal action is to move the robber
                    if action_parts[0] == "move" and action_parts[1] == "robber":

                        # Check that the robber is not being moved to the same tile
                        if (
                            robber_tile_q == int(action_parts[2])
                            and robber_tile_r == int(action_parts[3])
                            and robber_tile_s == int(action_parts[4])
                        ):
                            continue

                        else:
                            self.legal_actions[player_number].append(action)

                else:

                    # Does the player have enough resources to build a settlement?
                    if (
                        action_parts[0] == "build" and action_parts[1] == "settlement"
                    ) and (
                        self.resource_pool[player_number]["brick"] < 1
                        or self.resource_pool[player_number]["wool"] < 1
                        or self.resource_pool[player_number]["lumber"] < 1
                        or self.resource_pool[player_number]["grain"] < 1
                    ):
                        continue
                    # Does the player have enough resources to build a road?
                    elif (
                        action_parts[0] == "build" and action_parts[1] == "road"
                    ) and (
                        self.resource_pool[player_number]["brick"] < 1
                        or self.resource_pool[player_number]["lumber"] < 1
                    ):
                        continue
                    # Does the player have enough resources to build a city?
                    elif (
                        action_parts[0] == "build" and action_parts[1] == "city"
                    ) and (
                        self.resource_pool[player_number]["grain"] < 2
                        or self.resource_pool[player_number]["ore"] < 3
                    ):
                        continue
                    # Does the proposed settlement location already have a settlement on it?
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
                            # Is the settlement placement legal?
                            settlement_legal = (
                                self.board.check_settlement_placement_legal(
                                    q_coord,
                                    r_coord,
                                    s_coord,
                                    direction,
                                    player_number + 1,
                                )
                            )
                            if settlement_legal == False:
                                continue
                            else:
                                self.legal_actions[player_number].append(action)
                    # Does the proposed road location already have a road on it?
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
                            # Is the road placement legal?
                            road_legal = self.board.check_road_placement_legal(
                                q_coord, r_coord, s_coord, direction, player_number + 1
                            )
                            if road_legal == False:
                                continue
                            else:
                                self.legal_actions[player_number].append(action)
                    # Is the proposed city location already a city or an illegal build?
                    elif action_parts[0] == "build" and action_parts[1] == "city":
                        direction = action_parts[2]
                        q_coord = int(action_parts[3])
                        r_coord = int(action_parts[4])
                        s_coord = int(action_parts[5])
                        tile = self.board.get_tile(q_coord, r_coord, s_coord)
                        city_legal = self.board.check_city_placement_legal(
                            q_coord, r_coord, s_coord, direction, player_number + 1
                        )
                        if city_legal == True:
                            self.legal_actions[player_number].append(action)
                        else:
                            continue
                    # If the player tries to make a 4:1 trade with the bank, check if they have enough resources to make the trade
                    elif (
                        action_parts[0] == "trade"
                        and action_parts[1] == "bank"
                        and action_parts[2] == "4"
                        and action_parts[3] == "for"
                        and action_parts[4] == "1"
                    ):
                        if self.resource_pool[player_number][action_parts[5]] < 4:
                            continue
                        else:
                            self.legal_actions[player_number].append(action)
                    # At this stage, the action is legal, so add it to the list of legal actions
                    else:
                        # If the action doesn't contain the word "robber" it's legal
                        if "robber" not in action:
                            self.legal_actions[player_number].append(action)

    def get_legal_actions(self, player=1):
        # Return the list of legal actions
        return self.legal_actions[player - 1]

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
                f"build_city_northwest_{q_coord}_{r_coord}_{s_coord}"
            )
            self.all_actions.append(
                f"build_settlement_north_{q_coord}_{r_coord}_{s_coord}",
            )
            self.all_actions.append(
                f"build_city_north_{q_coord}_{r_coord}_{s_coord}",
            )
            self.all_actions.append(
                f"build_settlement_northeast_{q_coord}_{r_coord}_{s_coord}",
            )
            self.all_actions.append(
                f"build_city_northeast_{q_coord}_{r_coord}_{s_coord}",
            )
            self.all_actions.append(
                f"build_settlement_southeast_{q_coord}_{r_coord}_{s_coord}",
            )
            self.all_actions.append(
                f"build_city_southeast_{q_coord}_{r_coord}_{s_coord}",
            )
            self.all_actions.append(
                f"build_settlement_south_{q_coord}_{r_coord}_{s_coord}",
            )
            self.all_actions.append(
                f"build_city_south_{q_coord}_{r_coord}_{s_coord}",
            )
            self.all_actions.append(
                f"build_settlement_southwest_{q_coord}_{r_coord}_{s_coord}",
            )
            self.all_actions.append(
                f"build_city_southwest_{q_coord}_{r_coord}_{s_coord}",
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
            self.all_actions.append(f"move_robber_{q_coord}_{r_coord}_{s_coord}")

    def distribute_resources(self, roll):
        # Distribute resources to players based on the given dice roll
        if roll != 7:
            self.most_recent_resources_generated = 0
            tiles = self.board.get_board_tiles()
            for tile in tiles:
                if tile.get_tile_value() == roll and tile.get_has_robber() == False:

                    # Get the number of resources to give
                    occupied = tile.get_occupied_verticies()
                    for player in range(self.number_of_players):
                        resource_count = 0
                        for vertex in occupied:
                            if vertex == 1:
                                if vertex.get_owner() == player + 1:
                                    resource_count += 1
                            elif vertex == 2:
                                if vertex.get_owner() == player + 1:
                                    resource_count += 2
                        resource_to_give = tile.get_type()

                        # Give the correct resource to the player
                        self.resource_pool[player][resource_to_give] += resource_count

                    self.most_recent_resources_generated += resource_count

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
        total = dice_1 + dice_2
        if total == 7:
            self.robbery_in_progress = True
        return total

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
        self.resource_pool[0]["brick"] = 3
        self.resource_pool[0]["ore"] = 1
        self.resource_pool[0]["lumber"] = 2
        # Place the robber onto a tile
        self.board.move_robber(2, -1, -1)
        # Build a city for Player 2 at [0, 2, -2] north
        self.board.build_city(0, 2, -2, "north", 2)
        # Build a road for Player 2 at [0, 1, -1] east
        self.board.build_road(0, 1, -1, "east", 2)
        # Build a road for Player 2 at [0, 1, -1] southeast
        self.board.build_road(0, 1, -1, "southeast", 2)

    def get_game_over_flag(self):
        # Return the game over flag
        return self.game_over

    def start_game(self, number_of_players=1):
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

        # Place the robber onto the central desert tile
        self.board.move_robber(0, 0, 0)

        # Set legal actions
        self.set_legal_actions()

    def reward_information_request(self, action, legal_actions, player_number=1):
        # The Agent will request a batch of information about the current game state
        # Returns a dictionary of information

        information = {
            "legal_actions": legal_actions,
            "current_action": action,
            "game_over": self.game_over,
            "red_tiles": self.get_list_of_red_tile_coords(),
            "recent_resources_generated": self.most_recent_resources_generated,
        }

        return information

    def get_turn_number(self):
        # Return the current turn number
        return self.turn_number

    def get_list_of_red_tile_coords(self):
        # Return a list of the coordinates of all red tiles
        return self.board.get_list_of_red_tile_coords()

    def get_number_of_players(self):
        # Return the number of players
        return self.number_of_players
