# Unit test file for the Board class.

# # Imports
# import unittest
# from Board import Board
# from Tile import Tile
# import copy


# # Board dimension for the tests
# BOARD_DIMS = [3, 4, 5, 4, 3]

# # Board tiles for the tests
# generic_tile = Tile(type=1, board_tile_id=1, tile_value=4, has_robber=None, vert_northwest=None, vert_north=None, vert_northeast=None, vert_southeast=None, vert_south=None, vert_southwest=None, side_northwest=None, side_northeast=None, side_east=None, side_southeast=None, side_southwest=None, side_west=None, q_coord=None, r_coord=None, s_coord=None)
# BOARD_TILES = []
# for i in range(19):
#     BOARD_TILES.append(copy.copy(generic_tile))



# class TestBoardConstructor(unittest.TestCase):
#     def test_create_board(self):
#         # Test that a Board instance is created successfully
#         board = Board(BOARD_DIMS, BOARD_TILES)
#         self.assertIsInstance(board, Board)
    
#     def test_board_coordinates(self):
#         # Test that the board coordinates are set correctly
#         board = Board(BOARD_DIMS, BOARD_TILES)
#         self.assertEqual(board.get_board_tiles()[0].get_q_coord(), 0)
#         self.assertEqual(board.get_board_tiles()[5].get_r_coord(), -1)
#         self.assertEqual(board.get_board_tiles()[18].get_s_coord(), -2)
#         self.assertNotEqual(board.get_board_tiles()[9].get_q_coord(), 1)

# if __name__ == '__main__':
#     unittest.main()
