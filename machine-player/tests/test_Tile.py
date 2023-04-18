# !/usr/bin/env python3

# Unit test suite for the Tile class

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from Tile import Tile
from Road import Road
from Township import Township


class TestTile(unittest.TestCase):
    # Set the test tile class to be used in all tests
    def setUp(self):
        global test_tile
        test_tile = Tile(
            "wool",
            2,
            False,
            None,
            Township(),
            None,
            None,
            Township(),
            None,
            Road(),
            Road(),
            None,
            Road(),
            None,
            None,
            5,
            0,
            0,
            0,
        )

    def test_init(self):
        # Test that the tile is initilised correctly
        self.assertEqual(test_tile.get_type(), "wool")
        self.assertEqual(test_tile.get_board_tile_id(), 2)
        self.assertEqual(test_tile.get_has_robber(), False)
        self.assertEqual(test_tile.get_vert_northwest(), None)
        self.assertEqual(test_tile.get_vert_north(), 1)
        self.assertEqual(test_tile.get_vert_northeast(), None)
        self.assertEqual(test_tile.get_vert_southeast(), None)
        self.assertEqual(test_tile.get_vert_south(), 1)
        self.assertEqual(test_tile.get_vert_southwest(), None)
        self.assertEqual(test_tile.get_side_northwest(), 1)
        self.assertEqual(test_tile.get_side_northeast(), 1)
        self.assertEqual(test_tile.get_side_east(), None)
        self.assertEqual(test_tile.get_side_southeast(), 1)
        self.assertEqual(test_tile.get_side_southwest(), None)
        self.assertEqual(test_tile.get_side_west(), None)
        self.assertEqual(test_tile.get_tile_value(), 5)
        self.assertEqual(test_tile.get_q_coord(), 0)
        self.assertEqual(test_tile.get_r_coord(), 0)
        self.assertEqual(test_tile.get_s_coord(), 0)

    def test_get_state(self):
        # Test that the state is returned correctly
        self.assertEqual(
            test_tile.get_state(),
            {
                "type": "wool",
                "board_tile_id": 2,
                "has_robber": False,
                "vert_northwest": None,
                "vert_north": 1,
                "vert_northeast": None,
                "vert_southeast": None,
                "vert_south": 1,
                "vert_southwest": None,
                "side_northwest": 1,
                "side_northeast": 1,
                "side_east": None,
                "side_southeast": 1,
                "side_southwest": None,
                "side_west": None,
                "tile_value": 5,
                "q_coord": 0,
                "r_coord": 0,
                "s_coord": 0,
            },
        )

    def test_get_occupied_vertiicies(self):
        # Test that the occupied vertificies are returned correctly
        self.assertEqual(test_tile.get_occupied_verticies(), [1, 1])

    def test_get_vertex_from_direction(self):
        # Test that the correct vertex is returned
        self.assertEqual(test_tile.get_vertex_from_direction("north"), 1)
        self.assertEqual(test_tile.get_vertex_from_direction("northeast"), None)
        self.assertEqual(test_tile.get_vertex_from_direction("southeast"), None)
        self.assertEqual(test_tile.get_vertex_from_direction("south"), 1)
        self.assertEqual(test_tile.get_vertex_from_direction("southwest"), None)
        self.assertEqual(test_tile.get_vertex_from_direction("northwest"), None)
        self.assertEqual(test_tile.get_vertex_from_direction("rhubarb"), None)

    def test_build_settlement(self):
        # Test that a settlement can be built
        test_tile.build_settlement("southwest")
        self.assertEqual(test_tile.get_vert_southwest(), 1)
        self.assertEqual(test_tile.get_vert_north(), 1)
        self.assertEqual(test_tile.get_vert_south(), 1)
        self.assertEqual(test_tile.get_vert_northwest(), None)

    def test_build_city(self):
        # Test that a city can be built
        test_tile.build_city("south")
        self.assertEqual(test_tile.get_vert_south(), 2)
        self.assertEqual(test_tile.get_vert_north(), 1)
        self.assertEqual(test_tile.get_vert_southwest(), None)

    def test_get_side_from_direction(self):
        # Test that the correct side is returned
        self.assertEqual(test_tile.get_side_from_direction("northwest"), 1)
        self.assertEqual(test_tile.get_side_from_direction("northeast"), 1)
        self.assertEqual(test_tile.get_side_from_direction("east"), None)
        self.assertEqual(test_tile.get_side_from_direction("southeast"), 1)
        self.assertEqual(test_tile.get_side_from_direction("southwest"), None)
        self.assertEqual(test_tile.get_side_from_direction("west"), None)
        self.assertEqual(test_tile.get_side_from_direction("rhubarb"), None)

    def test_build_road(self):
        # Test that a road can be built
        test_tile.build_road("southwest")
        self.assertEqual(test_tile.get_side_southwest(), 1)
        self.assertEqual(test_tile.get_side_northwest(), 1)
        self.assertEqual(test_tile.get_side_east(), None)

    def test_get_neighbouring_verticies(self):
        # Test that the neighbouring verticies are returned correctly
        self.assertEqual(test_tile.get_neighbouring_verticies("northwest"), [1, None])

    def test_get_neighbouring_sides(self):
        # Test that the neighbouring sides are returned correctly
        self.assertEqual(test_tile.get_neighbouring_sides("northwest"), [1, None])

    def test_satisfies_distance_rule(self):
        # Test that distance rule legality is calculated correctly
        self.assertEqual(test_tile.satisfies_distance_rule("southwest"), False)
        test_tile.vert_south = None
        self.assertEqual(test_tile.satisfies_distance_rule("southwest"), True)

    def test_get_opposite_side_direction(self):
        # Test that the opposite side direction is returned correctly
        self.assertEqual(
            test_tile.get_opposite_side_direction("northwest"), "southeast"
        )
        self.assertEqual(
            test_tile.get_opposite_side_direction("southeast"), "northwest"
        )
        self.assertEqual(test_tile.get_opposite_side_direction("rhubarb"), None)

    def test_get_opposite_vertex_directions(self):
        # Test that the opposite vertex directions are returned correctly
        self.assertEqual(
            test_tile.get_opposite_vertex_directions("northwest"),
            ["northeast", "south"],
        )
        self.assertEqual(
            test_tile.get_opposite_vertex_directions("south"),
            ["northwest", "northeast"],
        )
        self.assertEqual(test_tile.get_opposite_vertex_directions("rhubarb"), None)

    def test_get_adjacent_sides_of_vertex(self):
        # Test that the adjacent sides of a vertex are returned correctly
        self.assertEqual(
            test_tile.get_adjacent_sides_of_vertex("northwest"), ["northwest", "west"]
        )
        self.assertEqual(
            test_tile.get_adjacent_sides_of_vertex("south"), ["southeast", "southwest"]
        )
        self.assertEqual(test_tile.get_adjacent_sides_of_vertex("rhubarb"), None)

    def test_is_vertex_adjacent_to_road(self):
        # Test that the correct boolean is returned
        self.assertEqual(test_tile.is_vertex_adjacent_to_road("north"), True)
        self.assertEqual(test_tile.is_vertex_adjacent_to_road("south"), True)
        self.assertEqual(test_tile.is_vertex_adjacent_to_road("southwest"), False)

    def test_get_all_side_values_as_list(self):
        # Test that the correct list of side values is returned
        self.assertEqual(
            test_tile.get_all_side_values_as_list(), [None, 1, None, None, 1, 1]
        )

    def test_get_all_vertex_values_as_list(self):
        # Test that the correct list of vertex values is returned
        self.assertEqual(
            test_tile.get_all_vertex_values_as_list(), [None, None, 1, None, None, 1]
        )

    def test_get_all_side_owners_as_list(self):
        # Test that the correct list of side owners is returned
        self.assertEqual(
            test_tile.get_all_side_owners_as_list(), [None, 0, None, None, 0, 0]
        )
        test_tile.side_east = Road(1)
        self.assertEqual(
            test_tile.get_all_side_owners_as_list(), [1, 0, None, None, 0, 0]
        )

    def test_get_all_vertex_owners_as_list(self):
        # Test that the correct list of vertex owners is returned
        self.assertEqual(
            test_tile.get_all_vertex_owners_as_list(), [None, None, 0, None, None, 0]
        )
