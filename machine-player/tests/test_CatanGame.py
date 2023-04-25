# !/usr/bin/env python3

# Unit test suite for the CatanGame class

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from CatanGame import CatanGame


class TestCatanGame(unittest.TestCase):
    def setUp(self):
        # Create a new game for each test
        self.game_started = CatanGame(number_of_players=2)
        # The game should be in progress
        self.game_started.setup_game_in_progress()

    def tearDown(self):
        # Delete the game after each test
        del self.game_started

    def test_init(self):
        # Test that reset() is called upon initialisation
        game = CatanGame()
        self.assertEqual(game.get_turn_number(), 0)

    def test_action_not_fulfilled_due_to_illegal_move(self):
        # Test that the state stays the same if the action is illegal

        # Get the value of the side at 0, -2, 2
        side = (
            self.game_started.get_board()
            .get_tile(0, -2, 2)
            .get_side_from_direction("northeast")
        )

        # Assert its value is 0
        self.assertEqual(side, None)

        # Try to build a road on the side
        self.game_started.step("build_road_northeast_0_-2_2")

        # Assert that the side is still 0
        self.assertEqual(side, None)

    def test_road_placement_fulfilled(self):
        # Test that the state changes if the action is legal

        # Get the value of the side at 0, -2, 2 (southeast)
        side = (
            self.game_started.get_board()
            .get_tile(-1, -1, 2)
            .get_side_from_direction("southeast")
        )

        # Assert its value is 0
        self.assertEqual(side, None)

        # Try to build a road on the side
        self.game_started.step("build_road_southeast_-1_-1_2")

        # Get the value of the side at 0, -2, 2 (southeast)
        side = (
            self.game_started.get_board()
            .get_tile(-1, -1, 2)
            .get_side_from_direction("southeast")
        )

        # Assert that the side is now 1
        self.assertEqual(side, 1)

    def test_settlement_placement_fulfilled(self):
        # Test that the state changes if the action is legal

        # Get the value of the vertex at 1, -2, 1 (southwest)
        town = (
            self.game_started.get_board()
            .get_tile(1, -2, 1)
            .get_vertex_from_direction("southwest")
        )

        # Assert its value is 0
        self.assertEqual(town, None)

        # Build a road first
        self.game_started.step("build_road_southeast_0_-2_2")

        # Try to build a settlement on the vertex
        self.game_started.step("build_settlement_southwest_1_-2_1")

        # Get the value of the vertex at 1, -2, 1 (southwest)
        town = (
            self.game_started.get_board()
            .get_tile(1, -2, 1)
            .get_vertex_from_direction("southwest")
        )

        # Get the value of the side at 0, -2, 2 (southeast)
        side = (
            self.game_started.get_board()
            .get_tile(0, -2, 2)
            .get_side_from_direction("southeast")
        )

        # Assert that the side is now 1
        self.assertEqual(side, 1)

        # Assert that the town is still None (because we don't have enough resources)
        self.assertEqual(town, None)

        # Give the player the resources to build a settlement
        self.game_started.resource_pool[0]["grain"] += 1
        self.game_started.resource_pool[0]["wool"] += 1

        # Set legal actions
        self.game_started.set_legal_actions()

        # Try to build a settlement on the vertex
        self.game_started.step("build_settlement_southwest_1_-2_1")

        # Get the value of the vertex at 1, -2, 1 (southwest)
        town = (
            self.game_started.get_board()
            .get_tile(1, -2, 1)
            .get_vertex_from_direction("southwest")
        )

        # Assert that the town is now 1
        self.assertEqual(town, 1)
