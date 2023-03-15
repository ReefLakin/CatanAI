"""
The class will contain the following attributes:
- side_tile_id (int)
- side_board_id (int)
- occupying_structure (which will be a new class Structure)

It will have the following methods:
- getters and setters for all the attributes
- a 'get_structure_info' method which takes no arguments and returns a dictionary
"""

class TileSide:
    def __init__(self, side_tile_id, side_board_id, occupying_structure):
        self.side_tile_id = side_tile_id
        self.side_board_id = side_board_id
        self.occupying_structure = occupying_structure

    def get_side_tile_id(self):
        return self.side_tile_id

    def get_side_board_id(self):
        return self.side_board_id

    def get_occupying_structure(self):
        return self.occupying_structure

    def set_side_tile_id(self, side_tile_id):
        self.side_tile_id = side_tile_id

    def set_side_board_id(self, side_board_id):
        self.side_board_id = side_board_id

    def set_occupying_structure(self, occupying_structure):
        self.occupying_structure = occupying_structure

    def get_structure_info(self):
        pass # Eventually make this a working method