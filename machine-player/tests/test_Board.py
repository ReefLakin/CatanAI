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
        global test_board, test_board_with_buildings
        test_board = Board(board_dims, tiles)
        test_board_with_buildings = Board(board_dims, tiles)
        test_board_with_buildings.build_settlement(0, 0, 0, "northwest")
        test_board_with_buildings.build_settlement(0, 2, -2, "southwest")
        test_board_with_buildings.build_road(0, 2, -2, "west")

    def test_get_board_dims(self):
        # Test that the board dimensions are returned correctly
        self.assertEqual(test_board.get_board_dims(), [3, 4, 5, 4, 3])
        new_board_dims = [4, 5, 6, 5, 4]
        test_board.board_dims = new_board_dims
        self.assertEqual(test_board.get_board_dims(), [4, 5, 6, 5, 4])

    def test_get_tile(self):
        # Test that the correct tile is being returned
        self.assertEqual(test_board.get_tile(0, 0, 0).get_type(), "desert")
        self.assertEqual(test_board.get_tile(1, 0, -1).get_tile_value(), 3)
        self.assertEqual(test_board.get_tile(2, 0, -2).get_type(), "ore")
        self.assertEqual(test_board.get_tile(0, 1, -1).get_type(), "grain")

    def test_shared_side_location(self):
        # Test that the correct tile is being returned
        test_tile_1 = test_board.shared_side_location(0, 0, 0, "northwest")
        test_tile_2 = test_board.shared_side_location(-1, -1, 2, "southwest")

        self.assertEqual(test_tile_1.get_tile_value(), 6)
        self.assertEqual(test_tile_2.get_tile_value(), 9)

    def test_shared_vertex_location(self):
        # Test that the correct tiles are being returned
        test_tiles_1 = test_board.shared_vertex_location(0, 0, 0, "north")
        test_tiles_2 = test_board.shared_vertex_location(-1, -1, 2, "south")

        self.assertEqual(test_tiles_1[0].get_tile_value(), 6)
        self.assertEqual(test_tiles_1[1].get_tile_value(), 4)
        self.assertEqual(test_tiles_2[0].get_tile_value(), 11)
        self.assertEqual(test_tiles_2[1].get_tile_value(), 9)

    def test_check_road_placement_legal(self):
        # Test that the road placement legal function is working correctly

        # Check False is returned on non-existant tiles
        self.assertFalse(
            test_board_with_buildings.check_road_placement_legal(3, 3, 3, "northwest")
        )

        # Check False is returned on an occupied side
        self.assertFalse(
            test_board_with_buildings.check_road_placement_legal(0, 2, -2, "west")
        )

        # Check True if the road placement is legal
        self.assertTrue(
            test_board_with_buildings.check_road_placement_legal(0, 2, -2, "northwest")
        )

        # Check False if it would be legal, but the player doens't own roads adjacent
        self.assertFalse(
            test_board_with_buildings.check_road_placement_legal(
                0, 2, -2, "northwest", 2
            )
        )

        # Check that the function works with adjacent tiles
        self.assertTrue(
            test_board_with_buildings.check_road_placement_legal(-1, 2, -1, "southeast")
        )
