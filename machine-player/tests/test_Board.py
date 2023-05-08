# !/usr/bin/env python3

# Unit test suite for the Board class

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from Board import Board
from Tile import Tile
from Township import Township
from Road import Road


class TestBoard(unittest.TestCase):
    # Create a board to be used for testing
    def setUp(self):
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
        a_different_set_of_tiles = []
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
            a_different_set_of_tiles.append(temp_tile)

        self.test_board = Board(board_dims, tiles)
        self.test_board_with_buildings = Board(board_dims, a_different_set_of_tiles)
        self.test_board_with_buildings.build_settlement(0, 0, 0, "northwest")
        self.test_board_with_buildings.build_settlement(0, 2, -2, "southwest")
        self.test_board_with_buildings.build_road(0, 2, -2, "west")
        self.test_board_with_buildings.build_settlement(0, -2, 2, "northeast")
        self.test_board_with_buildings.build_road(1, -2, 1, "northwest")
        self.test_board_with_buildings.build_road(1, -2, 1, "northeast")

    def tearDown(self):
        # Delete the board and the board with buildings
        del self.test_board
        del self.test_board_with_buildings

    def test_get_board_dims(self):
        # Test that the board dimensions are returned correctly
        self.assertEqual(self.test_board.get_board_dims(), [3, 4, 5, 4, 3])
        new_board_dims = [4, 5, 6, 5, 4]
        self.test_board.board_dims = new_board_dims
        self.assertEqual(self.test_board.get_board_dims(), [4, 5, 6, 5, 4])

    def test_get_tile(self):
        # Test that the correct tile is being returned
        self.assertEqual(self.test_board.get_tile(0, 0, 0).get_type(), "desert")
        self.assertEqual(self.test_board.get_tile(1, 0, -1).get_tile_value(), 3)
        self.assertEqual(self.test_board.get_tile(2, 0, -2).get_type(), "ore")
        self.assertEqual(self.test_board.get_tile(0, 1, -1).get_type(), "grain")

    def test_shared_side_location(self):
        # Test that the correct tile is being returned
        test_tile_1 = self.test_board.shared_side_location(0, 0, 0, "northwest")
        test_tile_2 = self.test_board.shared_side_location(-1, -1, 2, "southwest")

        self.assertEqual(test_tile_1.get_tile_value(), 6)
        self.assertEqual(test_tile_2.get_tile_value(), 9)

    def test_shared_vertex_location(self):
        # Test that the correct tiles are being returned
        test_tiles_1 = self.test_board.shared_vertex_location(0, 0, 0, "north")
        test_tiles_2 = self.test_board.shared_vertex_location(-1, -1, 2, "south")

        self.assertEqual(test_tiles_1[0].get_tile_value(), 6)
        self.assertEqual(test_tiles_1[1].get_tile_value(), 4)
        self.assertEqual(test_tiles_2[0].get_tile_value(), 11)
        self.assertEqual(test_tiles_2[1].get_tile_value(), 9)

    def test_check_road_placement_legal(self):
        # Test that the road placement legal function is working correctly

        # Check False is returned on non-existant tiles
        self.assertFalse(
            self.test_board_with_buildings.check_road_placement_legal(
                3, 3, 3, "northwest"
            )
        )

        # Check False is returned on an occupied side
        self.assertFalse(
            self.test_board_with_buildings.check_road_placement_legal(0, 2, -2, "west")
        )

        # Check True if the road placement is legal
        self.assertTrue(
            self.test_board_with_buildings.check_road_placement_legal(
                0, 2, -2, "northwest"
            )
        )

        # Check False if it would be legal, but the player doens't own roads adjacent
        self.assertFalse(
            self.test_board_with_buildings.check_road_placement_legal(
                0, 2, -2, "northwest", 2
            )
        )

        # Check that the function works with adjacent tiles
        self.assertTrue(
            self.test_board_with_buildings.check_road_placement_legal(
                -1, 2, -1, "southeast"
            )
        )

    def test_check_settlement_placement_legal(self):
        # Test that the settlement placement legal function is working correctly

        # Check False is returned on non-existant tiles
        self.assertFalse(
            self.test_board_with_buildings.check_settlement_placement_legal(
                3, 3, 3, "northwest"
            )
        )

        # Check False is returned on an occupied vertex
        self.assertFalse(
            self.test_board_with_buildings.check_settlement_placement_legal(
                0, 2, -2, "southwest"
            )
        )

        # Check False if the settlement placement breaks the distance rule
        self.assertFalse(
            self.test_board_with_buildings.check_settlement_placement_legal(
                0, 2, -2, "northwest"
            )
        )

        # Check False if it would be legal, but the player doens't own roads adjacent
        self.assertFalse(
            self.test_board_with_buildings.check_settlement_placement_legal(
                -1, 1, 0, "south"
            )
        )

        # Check True if settlement placement is legal
        self.assertTrue(
            self.test_board_with_buildings.check_settlement_placement_legal(
                1, -2, 1, "northeast"
            )
        )

        # Check this functionality works with adjacent tiles
        self.assertTrue(
            self.test_board_with_buildings.check_settlement_placement_legal(
                2, -2, 0, "northwest"
            )
        )

    def test_check_city_placement_legal(self):
        # Test that the city placement legal function is working correctly

        # Check False is returned on non-existant tiles
        self.assertFalse(
            self.test_board_with_buildings.check_city_placement_legal(
                3, 3, 3, "northwest"
            )
        )

        # Check False is returned on an unoccupied vertex
        self.assertFalse(
            self.test_board_with_buildings.check_city_placement_legal(0, 2, -2, "south")
        )

        # Check True is returned on an occupied vertex with a settlement
        self.assertTrue(
            self.test_board_with_buildings.check_city_placement_legal(
                0, 2, -2, "southwest"
            )
        )

        # Check False is returned if the occupied vertex is not owned by the player
        self.assertFalse(
            self.test_board_with_buildings.check_city_placement_legal(
                0, 2, -2, "southwest", 1
            )
        )

    def test_build_road(self):
        # Test that the road is being built correctly (and on all adjacent tiles)

        # Build a road
        self.test_board.build_road(1, 0, -1, "east", 2)

        # Check that the road is built on the correct tile
        self.assertEqual(
            self.test_board.get_tile(1, 0, -1)
            .get_side_from_direction("east")
            .get_owner(),
            2,
        )

        # Check that this works on the adjacent tile
        self.assertEqual(
            self.test_board.get_tile(2, 0, -2)
            .get_side_from_direction("west")
            .get_owner(),
            2,
        )

    def test_build_settlement(self):
        # Test that the settlement is being built correctly (and on all adjacent tiles)

        # Check that there is no settlement on the tile
        self.assertEqual(
            self.test_board.get_tile(1, 0, -1).get_vertex_from_direction("north"),
            None,
        )

        # Build a settlement
        self.test_board.build_settlement(1, 0, -1, "north", 2)

        # Check that the settlement is built on the correct tile
        self.assertEqual(
            self.test_board.get_tile(1, 0, -1)
            .get_vertex_from_direction("north")
            .get_owner(),
            2,
        )

        # Check that this works on the adjacent tiles
        self.assertEqual(
            self.test_board.get_tile(2, -1, -1)
            .get_vertex_from_direction("southwest")
            .get_owner(),
            2,
        )
        self.assertEqual(
            self.test_board.get_tile(1, -1, 0)
            .get_vertex_from_direction("southeast")
            .get_owner(),
            2,
        )

    def test_build_city(self):
        # Test that the city is being built correctly (and on all adjacent tiles)

        # Check that there is no city on the tile
        self.assertEqual(
            self.test_board.get_tile(1, 0, -1).get_vertex_from_direction("north"),
            None,
        )

        # Build a city
        self.test_board.build_city(1, 0, -1, "north", 2)

        # Check that the city is built on the correct tile
        self.assertEqual(
            self.test_board.get_tile(1, 0, -1)
            .get_vertex_from_direction("north")
            .get_owner(),
            2,
        )

        # Check that this works on the adjacent tiles
        self.assertEqual(
            self.test_board.get_tile(2, -1, -1)
            .get_vertex_from_direction("southwest")
            .get_owner(),
            2,
        )
        self.assertEqual(
            self.test_board.get_tile(1, -1, 0)
            .get_vertex_from_direction("southeast")
            .get_owner(),
            2,
        )

    def test_get_tile_types_in_a_list(self):
        # Test that the tile types are being returned correctly

        # Check that the correct tile types are returned
        self.assertEqual(
            self.test_board_with_buildings.get_tile_types_in_a_list(),
            [
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
            ],
        )

    def test_get_tile_numbers_in_a_list(self):
        # Test that the tile numbers are being returned correctly

        # Check that the correct tile numbers are returned
        self.assertEqual(
            self.test_board_with_buildings.get_tile_numbers_in_a_list(),
            [10, 2, 9, 12, 6, 4, 10, 9, 11, 0, 3, 8, 8, 3, 4, 5, 5, 6, 11],
        )

    def test_get_side_states(self):
        # Test that the side states are being returned correctly

        # Get side states of the presumed empty board
        side_states_empty = self.test_board.get_side_states()

        # Assert that there is not a 1 in the returned list
        self.assertFalse(1 in side_states_empty[0])

        # Place a road
        self.test_board.build_road(0, -2, 2, "east", 2)

        # Get side states of the board with a road
        side_states_road = self.test_board.get_side_states()

        # Assert that there is a 1 in the returned list
        self.assertTrue(1 in side_states_road[0])

    def test_get_vertex_states(self):
        # Test that the vertex states are being returned correctly

        # Get vertex states of the presumed empty board
        vertex_states_empty = self.test_board.get_vertex_states()

        # Assert that there is not a 1 in the returned list
        self.assertFalse(1 in vertex_states_empty[0])

        # Place a settlement
        self.test_board.build_settlement(0, -2, 2, "south", 2)

        # Get vertex states of the board with a settlement
        vertex_states_settlement = self.test_board.get_vertex_states()

        # Assert that there is a 1 in the returned list
        self.assertTrue(1 in vertex_states_settlement[0])

    def test_get_side_owners(self):
        # Test that the side owners are being returned correctly

        # Get side owners of the presumed empty board
        side_owners_empty = self.test_board.get_side_owners()

        # Assert that there is not a 1 in the returned list
        self.assertFalse(1 in side_owners_empty[0])

        # Place a road
        self.test_board.build_road(0, -2, 2, "east", 4)

        # Get side owners of the board with a road
        side_owners_road = self.test_board.get_side_owners()

        # Assert that there is a 4 in the returned list
        self.assertTrue(4 in side_owners_road[0])

    def test_get_vertex_owners(self):
        # Test that the vertex owners are being returned correctly

        # Get vertex owners of the presumed empty board
        vertex_owners_empty = self.test_board.get_vertex_owners()

        # Assert that there is not a 1 in the returned list
        self.assertFalse(1 in vertex_owners_empty[0])

        # Place a settlement
        self.test_board.build_settlement(0, -2, 2, "south", 4)

        # Get vertex owners of the board with a settlement
        vertex_owners_settlement = self.test_board.get_vertex_owners()

        # Assert that there is a 4 in the returned list
        self.assertTrue(4 in vertex_owners_settlement[0])

    def test_get_robber_tile(self):
        # Test that the robber tile is being returned correctly

        # Get the robber tile
        robber_tile = self.test_board_with_buildings.get_robber_tile()

        # Assert that at this stage the robber is on no tile
        self.assertEqual(robber_tile, None)

        # Move the robber
        self.test_board_with_buildings.move_robber(1, 0, -1)

        # Get the robber tile
        robber_tile = self.test_board_with_buildings.get_robber_tile()

        # Assert that the robber is on the correct tile
        self.assertEqual(robber_tile, self.test_board_with_buildings.get_tile(1, 0, -1))

    def test_move_robber(self):
        # Test that the robber is being moved correctly

        # Move the robber
        self.test_board_with_buildings.move_robber(1, 0, -1)

        # Get the robber tile
        robber_tile = self.test_board_with_buildings.get_robber_tile()

        # Assert that the robber is on the correct tile
        self.assertEqual(robber_tile, self.test_board_with_buildings.get_tile(1, 0, -1))

        # Move it again
        self.test_board_with_buildings.move_robber(2, -1, -1)

        # Get the robber tile
        robber_tile = self.test_board_with_buildings.get_robber_tile()

        # Assert that the robber is no longer on the old tile
        self.assertNotEqual(
            robber_tile, self.test_board_with_buildings.get_tile(1, 0, -1)
        )

    def test_get_robber_states(self):
        # Test that the robber states are being returned correctly

        # Get the robber states
        robber_states = self.test_board_with_buildings.get_robber_states()

        # Assert that the robber is not on any tile
        self.assertNotEqual(robber_states[0], 1)

        # Move the robber
        self.test_board_with_buildings.move_robber(0, -2, 2)

        # Get the robber states
        robber_states = self.test_board_with_buildings.get_robber_states()

        # Assert that the robber is on the correct tile
        self.assertEqual(robber_states[0], 1)

    def test_get_list_of_red_tile_coords(self):
        # Test that the list of red tile coords is being returned correctly

        # Get the list of red tile coords
        red_tile_coords = self.test_board_with_buildings.get_list_of_red_tile_coords()

        # Assert that the list is correct
        self.assertEqual(
            red_tile_coords, [[0, -1, 1], [2, 0, -2], [-2, 1, 1], [-1, 2, -1]]
        )

    def test_is_vertex_occupied(self):
        # Test that the vertex is being checked correctly

        # Check False if no settlement is present at the location
        self.assertFalse(
            self.test_board_with_buildings.is_vertex_occupied(0, 2, -2, "north")
        )

        # Check True if a settlement is present at the location
        self.assertTrue(
            self.test_board_with_buildings.is_vertex_occupied(0, 0, 0, "northwest")
        )

    def test_validate_distance_rule(self):
        # Test that the distance rule is being checked correctly

        # Check True if the distance rule is not violated
        self.assertTrue(
            self.test_board_with_buildings.validate_distance_rule(0, 2, -2, "north")
        )

        # Check False if the distance rule is not technically violated (even though a settlement already exists at the location)
        self.assertTrue(
            self.test_board_with_buildings.validate_distance_rule(0, 0, 0, "northwest")
        )

        # Check False if the distance rule is violated (due to a settlement being too close)
        self.assertFalse(
            self.test_board_with_buildings.validate_distance_rule(0, 0, 0, "southwest")
        )

        # Check False if the distance rule is violated (due to a settlement being too close, adj. tile)
        self.assertFalse(
            self.test_board_with_buildings.validate_distance_rule(-1, 0, 1, "north")
        )

    def test_build_random_adjacent_road(self):
        # Test that a road is being built randomly adjacent to a vertex correctly

        # Get side states
        side_states = self.test_board.get_side_states()

        # Assert that the road is not yet built
        self.assertNotIn(1, side_states[0])

        # Build a road randomly adjacent to the vertex
        self.test_board.build_random_adjacent_road(0, -2, 2, "north", 1)

        # Get side states
        side_states = self.test_board.get_side_states()

        # Assert that the road is built
        self.assertIn(1, side_states[0])
