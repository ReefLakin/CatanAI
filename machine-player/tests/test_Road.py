# !/usr/bin/env python3

# Unit test suite for the Road class
# The Road class file is kept one directory up from the tests directory

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from Road import Road


class TestRoad(unittest.TestCase):
    def test_init(self):
        # Test that the owner is set to 0 by default
        road = Road()
        self.assertEqual(road.get_owner(), 0)

        # Test that the owner can be set upon initialization
        road = Road(1)
        self.assertEqual(road.get_owner(), 1)

    def test_set_owner(self):
        # Test that the owner can be set
        road = Road()
        road.set_owner(1)
        self.assertEqual(road.get_owner(), 1)

    def test_eq(self):
        # Test that a road is equal to 1
        road = Road()
        self.assertEqual(road, 1)
        self.assertNotEqual(road, 2)
