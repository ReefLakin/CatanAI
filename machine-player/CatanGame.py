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
        self.game_phase = "main"
        board_dims = [3, 4, 5, 4, 3]
        self.pregame_build_turn_tracker = 0
        self.number_of_players = number_of_players
        # Get tile values
        tile_value_mode = "balanced"
        tile_values = self.generate_tile_values(tile_value_mode)
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
        self.victory_points = [0 for i in range(number_of_players)]
        # Keep a count of the road total for each player
        self.road_total = [0 for i in range(number_of_players)]
        # Set up the turn number
        self.turn_number = 0
        # Set the game over flag
        self.game_over = False
        # Set all possible actions
        self.set_all_possible_actions()
        # Set the legal actions
        self.set_legal_actions()
        # Set the dice roll
        self.most_recent_roll = (0, 0, 0, "Dice haven't been rolled yet.")
        # Set the number of resources generated with the last roll
        self.most_recent_resources_generated = 0
        # Set the most recent action
        self.most_recent_action = "No action has been taken yet."

    def get_board(self):
        # Return the game board
        return self.board

    def step(self, action, player_id=0):
        # Print the action taken to the console along with turn number
        print(
            "Turn "
            + str(self.turn_number)
            + ": "
            + action
            + "\nPlayer "
            + str(player_id)
            + "'s turn"
        )

        # Set the most recent action
        self.most_recent_action = action

        # Take a step in the game by applying the given action
        action_parts = action.split("_")

        # Is the action even legal?
        if action not in self.legal_actions[player_id]:
            pass

        # If the action is legal, is the action a 4:1 trade?
        elif action_parts[0] == "trade":
            # Reduce the traded resource by 4
            self.resource_pool[player_id][action_parts[5]] = (
                self.resource_pool[player_id][action_parts[5]] - 4
            )
            # Increase the received resource by 1
            self.resource_pool[player_id][action_parts[6]] = (
                self.resource_pool[player_id][action_parts[6]] + 1
            )

        # If the action is legal, is it a road build?
        elif action_parts[0] == "build" and action_parts[1] == "road":
            # Call a currently non-existent function to build a road
            self.board.build_road(
                int(action_parts[3]),
                int(action_parts[4]),
                int(action_parts[5]),
                action_parts[2],
                player_id,
            )

            # Increase the player's road total
            self.road_total[player_id] += 1

            # Remove 1 lumber and 1 brick from the resource pool
            # Unless the phase is "build"
            if self.game_phase != "build":
                self.resource_pool[player_id]["lumber"] = (
                    self.resource_pool[player_id]["lumber"] - 1
                )
                self.resource_pool[player_id]["brick"] = (
                    self.resource_pool[player_id]["brick"] - 1
                )

            # If it's the "build" phase, initial player tracker should increase
            if self.game_phase == "build":
                self.increase_pregame_player_tracker()

        # If the action is legal, is it a settlement build?
        elif action_parts[0] == "build" and action_parts[1] == "settlement":
            # Call a currently non-existent function to build a settlement
            self.board.build_settlement(
                int(action_parts[3]),
                int(action_parts[4]),
                int(action_parts[5]),
                action_parts[2],
                player_id,
            )

            # Update the player's VP count
            self.victory_points[player_id] += 1

            # Remove 1 lumber, 1 brick, 1 wool, and 1 grain from the resource pool
            # Unless the phase is "build"
            if self.game_phase != "build":
                self.resource_pool[player_id]["lumber"] = (
                    self.resource_pool[player_id]["lumber"] - 1
                )
                self.resource_pool[player_id]["brick"] = (
                    self.resource_pool[player_id]["brick"] - 1
                )
                self.resource_pool[player_id]["wool"] = (
                    self.resource_pool[player_id]["wool"] - 1
                )
                self.resource_pool[player_id]["grain"] = (
                    self.resource_pool[player_id]["grain"] - 1
                )

        # If the action is legal, is it a city build?
        elif action_parts[0] == "build" and action_parts[1] == "city":
            # Call a currently non-existent function to build a city
            self.board.build_city(
                int(action_parts[3]),
                int(action_parts[4]),
                int(action_parts[5]),
                action_parts[2],
                player_id,
            )

            # Update the player's VP count
            self.victory_points[player_id] += 1

            # Remove 2 grain and 3 ore from the resource pool
            self.resource_pool[player_id]["grain"] = (
                self.resource_pool[player_id]["grain"] - 2
            )
            self.resource_pool[player_id]["ore"] = (
                self.resource_pool[player_id]["ore"] - 3
            )

        # If the action is legal, is it a simple end turn?
        elif action == "end_turn":

            # Game phase should be 'main' if all players have placed their initial settlements
            if self.game_phase == "build":

                # Check everyone has at least two victory points
                if all(v >= 2 for v in self.victory_points):
                    # Check everyone has at least two roads
                    if all(r >= 2 for r in self.road_total):
                        # Set the game phase to 'main'
                        self.game_phase = "main"
                        # Set the turn number to 1
                        self.turn_number = 1

            else:

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

            # Set the game phase back to "main"
            self.game_phase = "main"

        # Is the game over?
        if self.victory_points[player_id] >= 10:
            self.game_over = True

        # Update the legal actions
        self.set_legal_actions()

    def get_state(self, player_id=0):
        # Build a dictionary of state information, which includes all of the tile state info, player resources, VPs and turn number.
        # This info is purposely in a human-readable state, but will likely be preprocessed before being fed to the DQN.
        return {
            "side_states": self.board.get_side_states(),
            "side_owners": self.board.get_side_owners(),
            "vertex_states": self.board.get_vertex_states(),
            "vertex_owners": self.board.get_vertex_owners(),
            "board_dims": self.board.get_board_dims(),
            "tile_types": self.board.get_tile_types_in_a_list(),
            "num_brick": self.resource_pool[player_id]["brick"],
            "num_lumber": self.resource_pool[player_id]["lumber"],
            "num_wool": self.resource_pool[player_id]["wool"],
            "num_grain": self.resource_pool[player_id]["grain"],
            "num_ore": self.resource_pool[player_id]["ore"],
            "victory_points": self.victory_points[player_id],
            "tile_values": self.board.get_tile_numbers_in_a_list(),
            "robber_states": self.board.get_robber_states(),
            "most_recent_roll": self.most_recent_roll[2],
        }

    def set_legal_actions(self):

        # Reset the legal actions list, which is a list of lists, one for each player
        self.legal_actions = [[] for i in range(self.number_of_players)]

        # Compile the list of legal actions for each player
        for player_id in range(self.number_of_players):

            # Here, we need to determine the current game "phase"

            # "main" is the main game phase, where players can build roads, settlements, cities, and end their turn
            if self.game_phase == "main":

                self.set_legal_actions_phase_main(player_id)

            # "robber" is the phase where the robber is in progress, and players can only move the robber
            elif self.game_phase == "robber":

                self.set_legal_actions_phase_robber(player_id)

            # "build" is the pre-game build phase, where players place their initial settlements and roads
            elif self.game_phase == "build":

                self.set_legal_actions_phase_build(player_id)

    def set_legal_actions_phase_main(self, player_id):

        # Iterate across all actions and split them into their parts
        for action in self.all_actions:
            action_parts = action.split("_")

            # Settlement Building
            if action_parts[0] == "build" and action_parts[1] == "settlement":

                # The player must have enough resources to build a settlement
                if (
                    self.resource_pool[player_id]["brick"] >= 1
                    and self.resource_pool[player_id]["wool"] >= 1
                    and self.resource_pool[player_id]["lumber"] >= 1
                    and self.resource_pool[player_id]["grain"] >= 1
                ):

                    # The proposed location must be empty
                    direction = action_parts[2]
                    q_coord = int(action_parts[3])
                    r_coord = int(action_parts[4])
                    s_coord = int(action_parts[5])
                    tile = self.board.get_tile(q_coord, r_coord, s_coord)
                    vert_val = tile.get_vertex_from_direction(direction)
                    if vert_val is None:

                        # The build must be legal
                        settlement_legal = self.board.check_settlement_placement_legal(
                            q_coord, r_coord, s_coord, direction, player_id
                        )

                        if settlement_legal == True:
                            self.legal_actions[player_id].append(action)

            # Road Building
            elif action_parts[0] == "build" and action_parts[1] == "road":

                # The player must have enough resources to build a road
                if (
                    self.resource_pool[player_id]["brick"] >= 1
                    and self.resource_pool[player_id]["lumber"] >= 1
                ):

                    # The proposed location must be empty
                    direction = action_parts[2]
                    q_coord = int(action_parts[3])
                    r_coord = int(action_parts[4])
                    s_coord = int(action_parts[5])
                    tile = self.board.get_tile(q_coord, r_coord, s_coord)
                    side_val = tile.get_side_from_direction(direction)
                    if side_val is None:

                        # The build must be legal
                        road_legal = self.board.check_road_placement_legal(
                            q_coord, r_coord, s_coord, direction, player_id
                        )

                        if road_legal == True:
                            self.legal_actions[player_id].append(action)

            # City Building
            elif action_parts[0] == "build" and action_parts[1] == "city":

                # The player must have enough resources to build a city
                if (
                    self.resource_pool[player_id]["ore"] >= 3
                    and self.resource_pool[player_id]["grain"] >= 2
                ):

                    # The build must be legal
                    direction = action_parts[2]
                    q_coord = int(action_parts[3])
                    r_coord = int(action_parts[4])
                    s_coord = int(action_parts[5])

                    city_legal = self.board.check_city_placement_legal(
                        q_coord, r_coord, s_coord, direction, player_id
                    )

                    if city_legal == True:
                        self.legal_actions[player_id].append(action)

            # 4:1 Trade with the Bank
            elif (
                action_parts[0] == "trade"
                and action_parts[1] == "bank"
                and action_parts[2] == "4"
                and action_parts[3] == "for"
                and action_parts[4] == "1"
            ):
                if self.resource_pool[player_id][action_parts[5]] >= 4:
                    self.legal_actions[player_id].append(action)

            # End Turn
            elif action_parts[0] == "end" and action_parts[1] == "turn":
                self.legal_actions[player_id].append(action)

    def set_legal_actions_phase_robber(self, player_id):

        # Iterate across all actions and split them into their parts
        for action in self.all_actions:
            action_parts = action.split("_")

            # Get the tile coordinates where the robber is currently located
            robber_tile = self.board.get_robber_tile()
            robber_tile_q = robber_tile.get_q_coord()
            robber_tile_r = robber_tile.get_r_coord()
            robber_tile_s = robber_tile.get_s_coord()

            # During the "robber" phase, the only legal action is to move the robber
            if action_parts[0] == "move" and action_parts[1] == "robber":

                # The robber can't be moved onto the same tile
                # Add all other tiles as potential spots for the robber to move to
                if (
                    robber_tile_q == int(action_parts[2])
                    and robber_tile_r == int(action_parts[3])
                    and robber_tile_s == int(action_parts[4])
                ) != True:
                    self.legal_actions[player_id].append(action)

    def set_legal_actions_phase_build(self, player_id):

        # Iterate across all actions and split them into their parts
        for action in self.all_actions:
            action_parts = action.split("_")

            victory_points = self.victory_points[player_id]
            road_total = self.road_total[player_id]

            # Settlement Building
            if action_parts[0] == "build" and action_parts[1] == "settlement":

                # The initial player tracker must equal the current player
                if self.pregame_build_turn_tracker == player_id:

                    # If the player has 0 victory points and 0 roads
                    # or if they have 1 victory point and 1 road, build a starting settlement
                    if (victory_points == 0 and road_total == 0) or (
                        victory_points == 1 and road_total == 1
                    ):

                        # The proposed location must be empty
                        direction = action_parts[2]
                        q_coord = int(action_parts[3])
                        r_coord = int(action_parts[4])
                        s_coord = int(action_parts[5])
                        tile = self.board.get_tile(q_coord, r_coord, s_coord)
                        vert_val = tile.get_vertex_from_direction(direction)
                        if vert_val is None:

                            # The build must be legal
                            settlement_legal = (
                                self.board.check_settlement_placement_legal(
                                    q_coord,
                                    r_coord,
                                    s_coord,
                                    direction,
                                    player_id,
                                    starting=True,
                                )
                            )

                            if settlement_legal == True:
                                self.legal_actions[player_id].append(action)

            # Road Building
            elif action_parts[0] == "build" and action_parts[1] == "road":

                # The initial player tracker must equal the current player
                if self.pregame_build_turn_tracker == player_id:

                    # If the player has 1 victory point and 0 roads,
                    # or if they have 2 victory points and 1 road, build a starting road
                    if (victory_points == 1 and road_total == 0) or (
                        victory_points == 2 and road_total == 1
                    ):

                        # The tile must be adjacent to the most recently placed settlement
                        direction = action_parts[2]
                        q_coord = int(action_parts[3])
                        r_coord = int(action_parts[4])
                        s_coord = int(action_parts[5])

                        last_split = self.most_recent_action.split("_")
                        last_direction = last_split[2]
                        last_q_coord = int(last_split[3])
                        last_r_coord = int(last_split[4])
                        last_s_coord = int(last_split[5])

                        acceptable_q_coords = [last_q_coord]
                        acceptable_r_coords = [last_r_coord]
                        acceptable_s_coords = [last_s_coord]

                        adjacent_tiles = self.board.shared_vertex_location(
                            last_q_coord, last_r_coord, last_s_coord, last_direction
                        )

                        if adjacent_tiles is not None:
                            for tile in adjacent_tiles:
                                if tile is not None:
                                    acceptable_q_coords.append(tile.get_q_coord())
                                    acceptable_r_coords.append(tile.get_r_coord())
                                    acceptable_s_coords.append(tile.get_s_coord())

                            if (
                                q_coord in acceptable_q_coords
                                and r_coord in acceptable_r_coords
                                and s_coord in acceptable_s_coords
                            ):

                                # The proposed location must be empty
                                tile = self.board.get_tile(q_coord, r_coord, s_coord)
                                side_val = tile.get_side_from_direction(direction)
                                if side_val is None:

                                    # The build must be legal
                                    road_legal = self.board.check_road_placement_legal(
                                        q_coord,
                                        r_coord,
                                        s_coord,
                                        direction,
                                        player_id,
                                        starting=True,
                                    )

                                    if road_legal == True:
                                        self.legal_actions[player_id].append(action)

            # Anything Else? (End Turn)
            else:
                if self.pregame_build_turn_tracker != player_id:
                    if action_parts[0] == "end" and action_parts[1] == "turn":
                        self.legal_actions[player_id].append(action)

    def get_legal_actions(self, player_id):
        # Return the list of legal actions
        return self.legal_actions[player_id]

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
                                if vertex.get_owner() == player:
                                    resource_count += 1
                            elif vertex == 2:
                                if vertex.get_owner() == player:
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
        # Print the roll to the console (let's turn that off for now)
        # print(f"Rolled a {dice_1} and a {dice_2} for a total of {dice_1 + dice_2}")
        # Store the roll in the game state
        self.most_recent_roll = (
            dice_1,
            dice_2,
            dice_1 + dice_2,
            f"{dice_1} and {dice_2}",
        )
        total = dice_1 + dice_2
        if total == 7:
            # 7 was rolled; set the game phase to "robber"
            self.game_phase = "robber"
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
        self.board.build_city(-1, 1, 0, "northwest")
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
        self.victory_points[0] = 3
        # Give the player 3 brick, 1 ore and 2 lumber
        self.resource_pool[0]["brick"] = 3
        self.resource_pool[0]["ore"] = 1
        self.resource_pool[0]["lumber"] = 2
        # Place the robber onto a tile
        self.board.move_robber(2, -1, -1)
        # Build a city for Player 2 at [0, 2, -2] north
        self.board.build_city(0, 2, -2, "north", 1)
        # Build a road for Player 2 at [0, 1, -1] east
        self.board.build_road(0, 1, -1, "east", 1)
        # Build a road for Player 2 at [0, 1, -1] southeast
        self.board.build_road(0, 1, -1, "southeast", 1)
        self.victory_points[1] = 2

    def get_game_over_flag(self):
        # Return the game over flag
        return self.game_over

    def start_game(self, number_of_players=1):

        # Reset the game state
        self.reset(number_of_players)

        # Set the game phase to "build"
        self.game_phase = "build"

        # Move the robber to the central desert tile
        self.board.move_robber(0, 0, 0)

        # Switch this on or off: settlements placed randomly at the start of the game
        PLACE_RANDOMLY = False
        NO_CHOICE_ALLOWED = False
        if PLACE_RANDOMLY:

            # Randomly build settlements and roads for all players
            for player in range(self.number_of_players):
                self.build_settlement_and_road_randomly(player)
            for player in range(self.number_of_players):
                self.build_settlement_and_road_randomly(player)

            # Set the game phase to "main"
            self.game_phase = "main"

        elif NO_CHOICE_ALLOWED:

            # For now, we're going to build for the player to start the game

            # We build in a fairly good spot for the first player:

            # Build a settlement at [1, -1, 0] northeast
            self.board.build_settlement(1, -1, 0, "northeast")
            # Build a road at [1, -1, 0] east
            self.board.build_road(1, -1, 0, "east")
            # Build a settlement at [-1, 1, 0] south
            self.board.build_settlement(-1, 1, 0, "south")
            # Build a road at [-1, 1, 0] southeast
            self.board.build_road(-1, 1, 0, "southeast")

            # Set the victory points to 2
            self.victory_points[0] = 2

            # Then we build for the second player if there is one

            if number_of_players > 1:
                self.number_of_players = number_of_players

                # Build a settlement at [-1, 0, 1] southwest
                self.board.build_settlement(-1, 0, 1, "southwest", 1)
                # Build a road at [-1, 0, 1] west
                self.board.build_road(-1, 0, 1, "west", 1)
                # Build a settlement at [0, -1, 1] north
                self.board.build_settlement(0, -1, 1, "north", 1)
                # Build a road at [0, -1, 1] northeast
                self.board.build_road(0, -1, 1, "northeast", 1)

                # Set the victory points to 2
                self.victory_points[1] = 2

            # Set the game phase to "main"
            self.game_phase = "main"

        # Set legal actions
        self.set_legal_actions()

    def reward_information_request(self, action, legal_actions):
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

    def get_player_id_of_current_winner(self):
        # Return the player ID of the current winner
        for i in range(self.number_of_players):
            if self.victory_points[i] >= 10:
                return i

    def build_settlement_and_road_randomly(self, player_id):

        # Function to support random starting positions at the beginning of the game
        build_complete = False

        while build_complete == False:

            rand_tile = None
            # Pick a random tile
            while rand_tile is None:
                rand_q = random.randint(-2, 2)
                rand_r = random.randint(-2, 2)
                rand_s = random.randint(-2, 2)
                rand_tile = self.board.get_tile(rand_q, rand_r, rand_s)

            # Pick a random vertex
            rand_direction = random.choice(
                [
                    "north",
                    "northeast",
                    "southeast",
                    "south",
                    "southwest",
                    "northwest",
                ]
            )

            # Check if the vertex is occupied
            vertex_occupied = self.board.is_vertex_occupied(
                rand_q, rand_r, rand_s, rand_direction
            )

            if vertex_occupied == False:

                # Is it a legal spot to build?
                distance_rule_validation = self.board.validate_distance_rule(
                    rand_q, rand_r, rand_s, rand_direction
                )

                if distance_rule_validation == True:

                    # Build the settlement
                    self.board.build_settlement(
                        rand_q, rand_r, rand_s, rand_direction, player_id
                    )

                    # Build an adjacent road
                    self.board.build_random_adjacent_road(
                        rand_q, rand_r, rand_s, rand_direction, player_id
                    )

                    # Add a road
                    self.road_total[player_id] += 1

                    # Add a victory point
                    self.victory_points[player_id] += 1

                    # Build complete
                    build_complete = True

    def increase_pregame_player_tracker(self):
        self.pregame_build_turn_tracker += 1
        if self.pregame_build_turn_tracker == self.number_of_players:
            self.pregame_build_turn_tracker = 0

    def generate_tile_values(self, mode):
        # Generate the values for each tile
        # mode = "vanilla", "balanced", "crazy"

        standard_list = [10, 2, 9, 12, 6, 4, 10, 9, 11, 0, 3, 8, 8, 3, 4, 5, 5, 6, 11]

        if mode == "vanilla":
            return standard_list

        elif mode == "balanced":
            # Shuffle the list
            random.shuffle(standard_list)
            # Ensure that 0 is always in position 9
            if standard_list.index(0) != 9:
                location_of_zero = standard_list.index(0)
                other_number = standard_list[9]
                standard_list[9] = 0
                standard_list[location_of_zero] = other_number
            return standard_list

        elif mode == "crazy":
            # Generate 19 random numbers between 2 and 12
            crazy_list = []
            for i in range(19):
                crazy_list.append(random.choice([2, 3, 4, 5, 6, 8, 9, 10, 11, 12]))
            crazy_list[9] = 0
            return crazy_list

        else:
            return standard_list
