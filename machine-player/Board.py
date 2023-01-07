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
        self.board_dims = board_dims # Array of ints, specfiying the dimensions of the board, e.g. [3, 4, 5, 4, 3]
        
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
            if tile.get_q_coord() == q and tile.get_r_coord() == r and tile.get_s_coord() == s:
                return tile