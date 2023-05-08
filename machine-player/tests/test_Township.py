# !/usr/bin/env python3

# Unit test suite for the Township class

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from Township import Township


class TestRoad(unittest.TestCase):
    def test_init(self):
        # Test that the owner and type are set to 0 and 1 by default, respectively
        town = Township()
        self.assertEqual(town.get_owner(), 0)
        self.assertEqual(town.get_type(), 1)

        # Test that the owner can be set upon initialisation
        town = Township(1)
        self.assertEqual(town.get_owner(), 1)

        # Test that the type can be set upon initialisation
        town = Township(0, 2)
        self.assertEqual(town.get_type(), 2)

    def test_set_owner(self):
        # Test that the owner can be set
        town = Township()
        town.set_owner(1)
        self.assertEqual(town.get_owner(), 1)

    def test_set_type(self):
        # Test that the type can be set
        town = Township()
        town.set_type(2)
        self.assertEqual(town.get_type(), 2)

    def test_eq(self):
        # Test that a town is equal to 1
        town = Township()
        self.assertEqual(town, 1)

        # Test that a city is equal to 2
        town = Township(0, 2)
        self.assertEqual(town, 2)
