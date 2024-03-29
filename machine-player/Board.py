"""
The class will contain the following attributes:
- board_dims (array of ints)
- board_tiles (array of Tile object arrays, each one representing a row, specified by board_dims)

It will have the following methods:
- getters for all the attributes
- a 'get_tile' method which takes two ints and returns a Tile object
"""

import Tile
import random


class Board:
    def __init__(self, board_dims, board_tiles):
        self.board_dims = board_dims  # Array of ints, specfiying the dimensions of the board, e.g. [3, 4, 5, 4, 3]

        # Set the q, r, s coordinates for each tile, if the board is a standard board
        if board_dims == [3, 4, 5, 4, 3]:
            q_vals = [0, 1, 2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, -2, -1, 0]
            r_vals = [-2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
            s_vals = [2, 1, 0, 2, 1, 0, -1, 2, 1, 0, -1, -2, 1, 0, -1, -2, 0, -1, -2]
            index = 0
            for tile in board_tiles:
                tile.set_q_coord(q_vals[index])
                tile.set_r_coord(r_vals[index])
                tile.set_s_coord(s_vals[index])
                index += 1
            self.board_tiles = board_tiles
        else:
            self.board_tiles = board_tiles

    # Getters
    def get_board_dims(self):
        return self.board_dims

    def get_board_tiles(self):
        return self.board_tiles

    # Get requested tile given q, r, s coordinates
    def get_tile(self, q, r, s):
        for tile in self.board_tiles:
            if (
                tile.get_q_coord() == q
                and tile.get_r_coord() == r
                and tile.get_s_coord() == s
            ):
                return tile
        return None

    # Get the tile that shares an edge with the given tile
    def shared_side_location(self, q, r, s, direction):
        if direction == "northwest":
            return self.get_tile(q, r - 1, s + 1)
        elif direction == "northeast":
            return self.get_tile(q + 1, r - 1, s)
        elif direction == "east":
            return self.get_tile(q + 1, r, s - 1)
        elif direction == "southeast":
            return self.get_tile(q, r + 1, s - 1)
        elif direction == "southwest":
            return self.get_tile(q - 1, r + 1, s)
        elif direction == "west":
            return self.get_tile(q - 1, r, s + 1)

    # Get the tiles that share a corner with the given tile
    def shared_vertex_location(self, q, r, s, direction):
        if direction == "northwest":
            return [
                self.shared_side_location(q, r, s, "west"),
                self.shared_side_location(q, r, s, "northwest"),
            ]
        elif direction == "north":
            return [
                self.shared_side_location(q, r, s, "northwest"),
                self.shared_side_location(q, r, s, "northeast"),
            ]
        elif direction == "northeast":
            return [
                self.shared_side_location(q, r, s, "northeast"),
                self.shared_side_location(q, r, s, "east"),
            ]
        elif direction == "southeast":
            return [
                self.shared_side_location(q, r, s, "east"),
                self.shared_side_location(q, r, s, "southeast"),
            ]
        elif direction == "south":
            return [
                self.shared_side_location(q, r, s, "southeast"),
                self.shared_side_location(q, r, s, "southwest"),
            ]
        elif direction == "southwest":
            return [
                self.shared_side_location(q, r, s, "southwest"),
                self.shared_side_location(q, r, s, "west"),
            ]

    # Can a player place a road on a given side location?
    def check_road_placement_legal(self, q, r, s, direction, player=0, starting=False):
        # Grab the tile
        tile = self.get_tile(q, r, s)
        # Check if the tile exists
        if tile == None:
            return False
        # Check if a road already exists on the side
        if tile.get_side_from_direction(direction) is not None:
            return False
        # If this is normal gameplay, following the following rules
        if starting == False:
            # Check that the side has at least one neighbouring road
            neighbouring_roads = tile.get_neighbouring_sides(
                direction, opp_blocking=True, player=player
            )
            for road in neighbouring_roads:
                if road == 1:
                    # Check that the road belongs to the player specified
                    if road.get_owner() == player:
                        return True
            else:
                # Check that the tile which shares the side has at least one neighbouring road
                neighbour_tile = self.shared_side_location(q, r, s, direction)
                if neighbour_tile == None:
                    return False
                else:
                    opposite_direction = tile.get_opposite_side_direction(direction)
                    neighbour_tile_neighbouring_roads = (
                        neighbour_tile.get_neighbouring_sides(
                            opposite_direction, opp_blocking=True, player=player
                        )
                    )
                    for road in neighbour_tile_neighbouring_roads:
                        if road == 1:
                            # Check that the road belongs to the player specified
                            if road.get_owner() == player:
                                return True
                    else:
                        return False
        # If this is the initial placement of roads, following the following rules
        else:
            # Check that the side has at least one neighbouring settlement
            if tile.is_side_adjacent_to_settlement(direction, player):
                return True
            else:
                return False

    # Can a player place a settlement on a given vertex location?
    def check_settlement_placement_legal(
        self, q, r, s, direction, player=0, starting=False
    ):
        # Grab the tile
        tile = self.get_tile(q, r, s)
        # Check if the tile exists
        if tile == None:
            return False
        # Check if a settlement already exists on the vertex
        if tile.get_vertex_from_direction(direction) is not None:
            return False
        # Does placing a settlement on this vertex break the distance rule? First, check the tile in question
        if tile.satisfies_distance_rule(direction) == False:
            return False
        # Continue checking against the distance rule. This time check the neighbouring tiles
        opposite_directions = tile.get_opposite_vertex_directions(direction)
        neighbouring_tiles = self.shared_vertex_location(q, r, s, direction)
        for i in range(2):
            if neighbouring_tiles[i] != None:
                if (
                    neighbouring_tiles[i].satisfies_distance_rule(
                        opposite_directions[i]
                    )
                    == False
                ):
                    return False
        # If 'starting' is true, we don't need to check that the settlement is connected to a road
        if starting == True:
            return True
        # Next, we have to check that the proposed settlement location is connected to a road. First, check the tile in question
        if tile.is_vertex_adjacent_to_road(direction, player) == True:
            return True
        # If that didn't come up true, all hope is not lost. Check the neighbouring tiles
        i = 0
        for tile in neighbouring_tiles:
            if tile != None:
                if (
                    tile.is_vertex_adjacent_to_road(opposite_directions[i], player)
                    == True
                ):
                    return True
            i += 1
        # If we've made it this far, the settlement is not connected to a road so it cannot be placed
        return False

    # Can a player place a city on a given vertex location?
    def check_city_placement_legal(self, q, r, s, direction, player=0):
        # Get the tile to start
        tile = self.get_tile(q, r, s)
        # Check if the tile exists
        if tile == None:
            return False
        # Check if a city already exists on the vertex
        if tile.get_vertex_from_direction(direction) == 2:
            return False
        # Check if a settlement exists on the vertex
        elif tile.get_vertex_from_direction(direction) == 1:
            # Check that the settlement belongs to the player specified
            if tile.get_vertex_from_direction(direction).get_owner() == player:
                return True
            else:
                return False
        else:
            return False

    # Build a road at a location, given the q, r, s coordinates and the direction
    def build_road(self, q, r, s, direction, player=0):
        # Build the road on the specified tile
        tile = self.get_tile(q, r, s)
        tile.build_road(direction, player)
        # Build a road on the neighbouring tile, if it exists
        neighbour_tile = self.shared_side_location(q, r, s, direction)
        if neighbour_tile != None:
            opposite_direction = tile.get_opposite_side_direction(direction)
            neighbour_tile.build_road(opposite_direction, player)

    # Build a settlement at a location, given the q, r, s coordinates and the direction
    def build_settlement(self, q, r, s, direction, player=0):
        # Build the settlement on the specified tile
        tile = self.get_tile(q, r, s)
        tile.build_settlement(direction, player)
        # Build a settlement on the neighbouring tiles, if they exist
        neighbour_tiles = self.shared_vertex_location(q, r, s, direction)
        direction_array = tile.get_opposite_vertex_directions(direction)
        for i in range(0, len(neighbour_tiles)):
            if neighbour_tiles[i] != None:
                neighbour_tiles[i].build_settlement(direction_array[i], player)

    # Build a city at a location, given the q, r, s coordinates and the direction
    def build_city(self, q, r, s, direction, player=0):
        # Build the city on the specified tile
        tile = self.get_tile(q, r, s)
        tile.build_city(direction, player)
        # Build a city on the neighbouring tiles, if they exist
        neighbour_tiles = self.shared_vertex_location(q, r, s, direction)
        direction_array = tile.get_opposite_vertex_directions(direction)
        for i in range(0, len(neighbour_tiles)):
            if neighbour_tiles[i] != None:
                neighbour_tiles[i].build_city(direction_array[i], player)

    # Get the tile types of all tiles
    def get_tile_types_in_a_list(self):
        tile_types = []
        for tile in self.board_tiles:
            tile_types.append(tile.get_type())
        return tile_types

    # Get the tile numbers of all tiles
    def get_tile_numbers_in_a_list(self):
        tile_values = []
        for tile in self.board_tiles:
            tile_values.append(tile.get_tile_value())
        return tile_values

    # Get the state of all edges on the board
    # None = no road, 1 = road
    # Edges that are shared by two tiles are listed twice
    def get_side_states(self):
        side_states = []
        for tile in self.board_tiles:
            side_states.append(tile.get_all_side_values_as_list())
        return side_states

    # Get the state of all vertices on the board
    # None = no settlement, 1 = settlement
    # Vertices that are shared by two or more tiles are listed possibly two or three times
    def get_vertex_states(self):
        vertex_states = []
        for tile in self.board_tiles:
            vertex_states.append(tile.get_all_vertex_values_as_list())
        return vertex_states

    # Get the owners of all the edges (roads) on the board
    # None = no road, 1 = player 1, 2 = player 2, etc.
    # Edges that are shared by two tiles are listed twice
    def get_side_owners(self):
        side_owners = []
        for tile in self.board_tiles:
            side_owners.append(tile.get_all_side_owners_as_list())
        return side_owners

    # Get the owners of all the vertices (settlements and cities) on the board
    # None = no settlement, 1 = player 1, 2 = player 2, etc.
    # Vertices that are shared by two or more tiles are listed possibly two or three times
    def get_vertex_owners(self):
        vertex_owners = []
        for tile in self.board_tiles:
            vertex_owners.append(tile.get_all_vertex_owners_as_list())
        return vertex_owners

    # Get the tile currently holding the robber
    def get_robber_tile(self):
        for tile in self.board_tiles:
            if tile.get_has_robber() == True:
                return tile
        return None

    # Move the robber from current robber tile to a new tile
    def move_robber(self, q, r, s):
        # Get the current robber tile
        current_robber_tile = self.get_robber_tile()
        # Move the robber to the new tile
        new_robber_tile = self.get_tile(q, r, s)
        if current_robber_tile != None:
            current_robber_tile.set_has_robber(False)
        if new_robber_tile != None:
            new_robber_tile.set_has_robber(True)

    # Get a list of tiles (0 if there isn't a robber on the tile, 1 if there is)
    def get_robber_states(self):
        robber_states = []
        for tile in self.board_tiles:
            if tile.get_has_robber() == True:
                robber_states.append(1)
            else:
                robber_states.append(0)
        return robber_states

    # Get a list of all tile coords representing red tiles
    def get_list_of_red_tile_coords(self):
        red_tile_coords = []
        for tile in self.board_tiles:
            if tile.get_tile_value() == 6 or tile.get_tile_value() == 8:
                red_tile_coords.append(
                    [tile.get_q_coord(), tile.get_r_coord(), tile.get_s_coord()]
                )
        return red_tile_coords

    # Given a tile and a direction, return if that tile is occupied already
    def is_vertex_occupied(self, q, r, s, direction):
        tile = self.get_tile(q, r, s)
        if tile.get_vertex_from_direction(direction) != None:
            return True
        else:
            return False

    # Make a distance rule check
    def validate_distance_rule(self, q, r, s, direction):
        # Get the tile
        tile = self.get_tile(q, r, s)

        # Check if the tile exists
        if tile == None:
            return False

        # Check on the immediately tile if the distance rule is violated
        if tile.satisfies_distance_rule(direction) == False:
            return False

        # Then check on the neighbouring tiles
        opposite_directions = tile.get_opposite_vertex_directions(direction)
        neighbouring_tiles = self.shared_vertex_location(q, r, s, direction)
        for i in range(2):
            if neighbouring_tiles[i] != None:
                if (
                    neighbouring_tiles[i].satisfies_distance_rule(
                        opposite_directions[i]
                    )
                    == False
                ):
                    return False

        # If we get here, the distance rule is satisfied
        return True

    # Build a road randomly adjacent to a vertex location
    def build_random_adjacent_road(self, q, r, s, direction, player):
        # Get the tile
        tile = self.get_tile(q, r, s)

        # Get the adjacent sides
        direction_possible = tile.get_adjacent_sides_of_vertex(direction)

        # Shuffle the direction possibilities
        random.shuffle(direction_possible)

        if tile.get_side_from_direction(direction_possible[0]) == None:
            self.build_road(q, r, s, direction_possible[0], player)
        else:
            self.build_road(q, r, s, direction_possible[1], player)
