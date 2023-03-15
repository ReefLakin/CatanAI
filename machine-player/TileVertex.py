"""
The class will contain the following attributes:
- vertex_tile_id (int)
- vertex_board_id (int)
- occupying_structure (which will be a new class Structure)

It will have the following methods:
- getters and setters for all the attributes
- a 'get_structure_info' method which takes no arguments and returns a dictionary
"""

class TileVertex:
    def __init__(self, vertex_tile_id, vertex_board_id, occupying_structure):
        self.vertex_tile_id = vertex_tile_id
        self.vertex_board_id = vertex_board_id
        self.occupying_structure = occupying_structure

    def get_vertex_tile_id(self):
        return self.vertex_tile_id

    def get_vertex_board_id(self):
        return self.vertex_board_id

    def get_occupying_structure(self):
        return self.occupying_structure

    def set_vertex_tile_id(self, vertex_tile_id):
        self.vertex_tile_id = vertex_tile_id

    def set_vertex_board_id(self, vertex_board_id):
        self.vertex_board_id = vertex_board_id

    def set_occupying_structure(self, occupying_structure):
        self.occupying_structure = occupying_structure

    def get_structure_info(self):
        pass # Eventually make this a working method