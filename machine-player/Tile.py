"""
The class will contain the following attributes:
- type (the type of which will be another class TileType)
- board_tile_id (int)
- has_robber (boolean)
- vertex_1 (the type of which will be another class TileVertex)
- vertex_2
- vertex_3
- vertex_4
- vertex_5
- vertex_6
- side_1 (the type of which will be a new class TileSide)
- side_2
- side_3
- side_4
- side_5
- side_6
- tile_value (int)

It will have the following methods:
- getters and setters for all the attributes
- an 'is_neighbouring_vertex' method which takes two ints and returns a bool
- an 'is_neighbouring_side' method which takes two ints and returns a bool
"""

class Tile:
    def __init__(self, type, board_tile_id, has_robber, vertex_1, vertex_2, vertex_3, vertex_4, vertex_5, vertex_6, side_1, side_2, side_3, side_4, side_5, side_6, tile_value):
        self.type = type
        self.board_tile_id = board_tile_id
        self.has_robber = has_robber
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        self.vertex_3 = vertex_3
        self.vertex_4 = vertex_4
        self.vertex_5 = vertex_5
        self.vertex_6 = vertex_6
        self.side_1 = side_1
        self.side_2 = side_2
        self.side_3 = side_3
        self.side_4 = side_4
        self.side_5 = side_5
        self.side_6 = side_6
        self.tile_value = tile_value

    def get_type(self):
        return self.type

    def set_type(self, type):
        self.type = type

    def get_board_tile_id(self):
        return self.board_tile_id

    def set_board_tile_id(self, board_tile_id):
        self.board_tile_id = board_tile_id

    def get_has_robber(self):
        return self.has_robber

    def set_has_robber(self, has_robber):
        self.has_robber = has_robber

    def get_vertex_1(self):
        return self.vertex_1

    def set_vertex_1(self, vertex_1):
        self.vertex_1 = vertex_1

    def get_vertex_2(self):
        return self.vertex_2

    def set_vertex_2(self, vertex_2):
        self.vertex_2 = vertex_2

    def get_vertex_3(self):
        return self.vertex_3

    def set_vertex_3(self, vertex_3):
        self.vertex_3 = vertex_3

    def get_vertex_4(self):
        return self.vertex_4

    def set_vertex_4(self, vertex_4):
        self.vertex_4 = vertex_4

    def get_vertex_5(self):
        return self.vertex_5

    def set_vertex_5(self, vertex_5):
        self.vertex_5 = vertex_5
    
    def get_vertex_6(self):
        return self.vertex_6
    
    def set_vertex_6(self, vertex_6):
        self.vertex_6 = vertex_6
    
    def get_side_1(self):
        return self.side_1
    
    def set_side_1(self, side_1):
        self.side_1 = side_1
    
    def get_side_2(self):
        return self.side_2
    
    def set_side_2(self, side_2):
        self.side_2 = side_2

    def get_side_3(self):
        return self.side_3
    
    def set_side_3(self, side_3):
        self.side_3 = side_3
    
    def get_side_4(self):
        return self.side_4

    def set_side_4(self, side_4):
        self.side_4 = side_4
    
    def get_side_5(self):
        return self.side_5
    
    def set_side_5(self, side_5):
        self.side_5 = side_5
    
    def get_side_6(self):
        return self.side_6
    
    def set_side_6(self, side_6):
        self.side_6 = side_6
    
    def get_tile_value(self):
        return self.tile_value
    
    def set_tile_value(self, tile_value):
        self.tile_value = tile_value
    
    def is_neighbouring_vertex(self, vertex_1, vertex_2):
        pass

    def is_neighbouring_side(self, side_1, side_2):
        pass

# To do: Add type hints to getters and setters
