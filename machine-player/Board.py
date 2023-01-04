"""
The class will contain the following attributes:
- board_dims (array of ints)
- board_tiles (array of Tile object arrays, each one representing a row, specified by board_dims)

It will have the following methods:
- getters for all the attributes
- a 'get_tile' method which takes two ints and returns a Tile object
"""

class Board:
    def __init__(self, board_dims, board_tiles):
        self.board_dims = board_dims
        self.board_tiles = board_tiles

    def get_board_dims(self):
        return self.board_dims

    def get_board_tiles(self):
        return self.board_tiles

    def get__tile(self, row, col):
        return self.board_tiles[row][col]