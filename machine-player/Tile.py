"""
The class will contain the following attributes:
- type (the type of which will be another class TileType)
- board_tile_id (int)
- has_robber (boolean)
- vert_northwest (the type of which will be another class TileVertex)
- vert_north
- vert_northeast
- vert_southeast
- vert_south
- vert_southwest
- side_northwest (the type of which will be a new class TileSide)
- side_northeast
- side_east
- side_southeast
- side_southwest
- side_west
- tile_value (int)
- q_coord (int)
- r_coord (int)
- s_coord (int)

It will have the following methods:
- getters and setters for all the attributes
- an 'is_neighbouring_vertex' method which takes two ints and returns a bool
- an 'is_neighbouring_side' method which takes two ints and returns a bool
"""

class Tile:
    # Constructor
    def __init__(self, type, board_tile_id, has_robber, vert_northwest, vert_north, vert_northeast, vert_southeast, vert_south, vert_southwest, side_northwest, side_northeast, side_east, side_southeast, side_southwest, side_west, tile_value, q_coord, r_coord, s_coord):
        self.type = type
        self.board_tile_id = board_tile_id
        self.has_robber = has_robber
        self.vert_northwest = vert_northwest
        self.vert_north = vert_north
        self.vert_northeast = vert_northeast
        self.vert_southeast = vert_southeast
        self.vert_south = vert_south
        self.vert_southwest = vert_southwest
        self.side_northwest = side_northwest
        self.side_northeast = side_northeast
        self.side_east = side_east
        self.side_southeast = side_southeast
        self.side_southwest = side_southwest
        self.side_west = side_west
        self.tile_value = tile_value
        self.q_coord = q_coord
        self.r_coord = r_coord
        self.s_coord = s_coord
    
    # Alternative contructor, that only takes the type and board_tile_id; the rest of the attributes are set to default values
    @classmethod
    def from_type_and_id(cls, type, board_tile_id):
        return cls(type, board_tile_id, False, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)


    # Getters and setters
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

    def get_vert_northwest(self):
        return self.vert_northwest

    def set_vert_northwest(self, vert_northwest):
        self.vert_northwest = vert_northwest

    def get_vert_north(self):
        return self.vert_north

    def set_vert_north(self, vert_north):
        self.vert_north = vert_north

    def get_vert_northeast(self):
        return self.vert_northeast

    def set_vert_northeast(self, vert_northeast):
        self.vert_northeast = vert_northeast

    def get_vert_southeast(self):
        return self.vert_southeast

    def set_vert_southeast(self, vert_southeast):
        self.vert_southeast = vert_southeast

    def get_vert_south(self):
        return self.vert_south

    def set_vert_south(self, vert_south):
        self.vert_south = vert_south
    
    def get_vert_southwest(self):
        return self.vert_southwest
    
    def set_vert_southwest(self, vert_southwest):
        self.vert_southwest = vert_southwest
    
    def get_side_northwest(self):
        return self.side_northwest
    
    def set_side_northwest(self, side_northwest):
        self.side_northwest = side_northwest
    
    def get_side_northeast(self):
        return self.side_northeast
    
    def set_side_northeast(self, side_northeast):
        self.side_northeast = side_northeast

    def get_side_east(self):
        return self.side_east
    
    def set_side_east(self, side_east):
        self.side_east = side_east
    
    def get_side_southeast(self):
        return self.side_southeast

    def set_side_southeast(self, side_southeast):
        self.side_southeast = side_southeast
    
    def get_side_southwest(self):
        return self.side_southwest
    
    def set_side_southwest(self, side_southwest):
        self.side_southwest = side_southwest
    
    def get_side_west(self):
        return self.side_west
    
    def set_side_west(self, side_west):
        self.side_west = side_west
    
    def get_tile_value(self):
        return self.tile_value
    
    def set_tile_value(self, tile_value):
        self.tile_value = tile_value

    def get_q_coord(self):
        return self.q_coord

    def set_q_coord(self, q_coord):
        self.q_coord = q_coord
    
    def get_r_coord(self):
        return self.r_coord
    
    def set_r_coord(self, r_coord):
        self.r_coord = r_coord
    
    def get_s_coord(self):
        return self.s_coord
    
    def set_s_coord(self, s_coord):
        self.s_coord = s_coord

    # Other methods
    
    # If two vertices are next to each other on the board, return True. Otherwise, return False.
    def is_neighbouring_vertex(self, vert_northwest, vert_north):
        
        # Create array of all vertex IDs from 1-6.
        vertex_ids = [self.vert_northwest.get_vertex_id(), self.vert_north.get_vertex_id(), self.vert_northeast.get_vertex_id(), self.vert_southeast.get_vertex_id(), self.vert_south.get_vertex_id(), self.vert_southwest.get_vertex_id()]

        # If vert_northwest isn't in the array, return False.
        if vert_northwest not in vertex_ids:
            return False
        
        # If vert_north isn't in the array, return False.
        if vert_north not in vertex_ids:
            return False
        
        # If vert_northwest and vert_north are next to each other in the array, return True.
        index_to_check_lower = (vertex_ids.index(vert_northwest) - 1)
        index_to_check_higher = (vertex_ids.index(vert_northwest) + 1)
        if index_to_check_lower < 0:
            index_to_check_lower = 5
        if index_to_check_higher > 5:
            index_to_check_higher = 0
        if vertex_ids[index_to_check_lower] == vert_north or vertex_ids[index_to_check_higher] == vert_north:
            return True
        
        return False

    # If two sides are next to each other on the board, return True. Otherwise, return False.
    def is_neighbouring_side(self, side_northwest, side_northeast):
            
            # Create array of all side IDs from 1-6.
            side_ids = [self.side_northwest.get_side_id(), self.side_northeast.get_side_id(), self.side_east.get_side_id(), self.side_southeast.get_side_id(), self.side_southwest.get_side_id(), self.side_west.get_side_id()]
    
            # If side_northwest isn't in the array, return False.
            if side_northwest not in side_ids:
                return False
            
            # If side_northeast isn't in the array, return False.
            if side_northeast not in side_ids:
                return False
            
            # If side_northwest and side_northeast are next to each other in the array, return True.
            index_to_check_lower = (side_ids.index(side_northwest) - 1)
            index_to_check_higher = (side_ids.index(side_northwest) + 1)
            if index_to_check_lower < 0:
                index_to_check_lower = 5
            if index_to_check_higher > 5:
                index_to_check_higher = 0
            if side_ids[index_to_check_lower] == side_northeast or side_ids[index_to_check_higher] == side_northeast:
                return True
            
            return False

# To do: Add type hints to getters and setters
