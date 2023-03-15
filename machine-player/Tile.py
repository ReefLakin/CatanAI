"""
The class will contain the following attributes:
- type (the type of which will be another class TileType, but for now it will be a string)
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
    def __init__(
        self,
        type,
        board_tile_id,
        has_robber,
        vert_northwest,
        vert_north,
        vert_northeast,
        vert_southeast,
        vert_south,
        vert_southwest,
        side_northwest,
        side_northeast,
        side_east,
        side_southeast,
        side_southwest,
        side_west,
        tile_value,
        q_coord,
        r_coord,
        s_coord,
    ):
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

    # Return the state of the entire tile.
    # The state will be of type dict.
    def get_state(self):
        return {
            "type": self.type,
            "board_tile_id": self.board_tile_id,
            "has_robber": self.has_robber,
            "vert_northwest": self.vert_northwest,
            "vert_north": self.vert_north,
            "vert_northeast": self.vert_northeast,
            "vert_southeast": self.vert_southeast,
            "vert_south": self.vert_south,
            "vert_southwest": self.vert_southwest,
            "side_northwest": self.side_northwest,
            "side_northeast": self.side_northeast,
            "side_east": self.side_east,
            "side_southeast": self.side_southeast,
            "side_southwest": self.side_southwest,
            "side_west": self.side_west,
            "tile_value": self.tile_value,
            "q_coord": self.q_coord,
            "r_coord": self.r_coord,
            "s_coord": self.s_coord,
        }

    # Given two

    # Return a list of all the tiles that are not equal to None
    def get_occupied_verticies(self):

        occupied_veriticies = []

        if self.vert_northwest != None:
            occupied_veriticies.append(self.vert_northwest)
        if self.vert_southwest != None:
            occupied_veriticies.append(self.vert_southwest)
        if self.vert_south != None:
            occupied_veriticies.append(self.vert_south)
        if self.vert_southeast != None:
            occupied_veriticies.append(self.vert_southeast)
        if self.vert_northeast != None:
            occupied_veriticies.append(self.vert_northeast)
        if self.vert_north != None:
            occupied_veriticies.append(self.vert_north)

        return occupied_veriticies

    # Return the value of a specified vertex
    def get_vertex_from_direction(self, direction):
        if direction == "northwest":
            return self.vert_northwest
        elif direction == "north":
            return self.vert_north
        elif direction == "northeast":
            return self.vert_northeast
        elif direction == "southeast":
            return self.vert_southeast
        elif direction == "south":
            return self.vert_south
        elif direction == "southwest":
            return self.vert_southwest
        else:
            return None

    # Set the value of a specified vertex to 1 (settlement)
    def build_settlement(self, direction):
        if direction == "northwest":
            self.vert_northwest = 1
        elif direction == "north":
            self.vert_north = 1
        elif direction == "northeast":
            self.vert_northeast = 1
        elif direction == "southeast":
            self.vert_southeast = 1
        elif direction == "south":
            self.vert_south = 1
        elif direction == "southwest":
            self.vert_southwest = 1
        else:
            return None

    # Set the value of a specified vertex to 2 (city)
    def build_city(self, direction):
        if direction == "northwest":
            self.vert_northwest = 2
        elif direction == "north":
            self.vert_north = 2
        elif direction == "northeast":
            self.vert_northeast = 2
        elif direction == "southeast":
            self.vert_southeast = 2
        elif direction == "south":
            self.vert_south = 2
        elif direction == "southwest":
            self.vert_southwest = 2
        else:
            return None

    # Return the value of a specified side
    def get_side_from_direction(self, direction):
        if direction == "northwest":
            return self.side_northwest
        elif direction == "northeast":
            return self.side_northeast
        elif direction == "east":
            return self.side_east
        elif direction == "southeast":
            return self.side_southeast
        elif direction == "southwest":
            return self.side_southwest
        elif direction == "west":
            return self.side_west
        else:
            return None

    # Set the value of a specified side to 1 (road)
    def build_road(self, direction):
        if direction == "northwest":
            self.side_northwest = 1
        elif direction == "northeast":
            self.side_northeast = 1
        elif direction == "east":
            self.side_east = 1
        elif direction == "southeast":
            self.side_southeast = 1
        elif direction == "southwest":
            self.side_southwest = 1
        elif direction == "west":
            self.side_west = 1

    # Return the neighbouring verticies of a specified vertex
    def get_neighbouring_verticies(self, direction):
        if direction == "northwest":
            return [self.vert_north, self.vert_southwest]
        elif direction == "north":
            return [self.vert_northwest, self.vert_northeast]
        elif direction == "northeast":
            return [self.vert_north, self.vert_southeast]
        elif direction == "southeast":
            return [self.vert_south, self.vert_northeast]
        elif direction == "south":
            return [self.vert_southeast, self.vert_southwest]
        elif direction == "southwest":
            return [self.vert_south, self.vert_northwest]
        else:
            return None

    # Return the neighbouring sides of a specified side
    def get_neighbouring_sides(self, direction):
        if direction == "northwest":
            return [self.side_northeast, self.side_west]
        elif direction == "northeast":
            return [self.side_northwest, self.side_east]
        elif direction == "east":
            return [self.side_northeast, self.side_southeast]
        elif direction == "southeast":
            return [self.side_east, self.side_southwest]
        elif direction == "southwest":
            return [self.side_southeast, self.side_west]
        elif direction == "west":
            return [self.side_southwest, self.side_northwest]
        else:
            return None

    # Return a bool indicating whether or not placing on a specific vertex satisfies the distance rule
    def satisfies_distance_rule(self, direction):
        neighbours = self.get_neighbouring_verticies(direction)
        for neighbour in neighbours:
            if neighbour == 1 or neighbour == 2:
                return False
        return True

    # Return the opposite side direction of a specified direction
    def get_opposite_side_direction(self, direction):
        if direction == "northwest":
            return "southeast"
        elif direction == "northeast":
            return "southwest"
        elif direction == "east":
            return "west"
        elif direction == "southeast":
            return "northwest"
        elif direction == "southwest":
            return "northeast"
        elif direction == "west":
            return "east"
        else:
            return None

    # Return the 2 opposite vertex directions of a specified direction
    # This needs to be in a clockwise order to work with higher level functions; possibly change this later
    def get_opposite_vertex_directions(self, direction):
        if direction == "northwest":
            return ["northeast", "south"]
        elif direction == "north":
            return ["southeast", "southwest"]
        elif direction == "northeast":
            return ["south", "northwest"]
        elif direction == "southeast":
            return ["southwest", "north"]
        elif direction == "south":
            return ["northwest", "northeast"]
        elif direction == "southwest":
            return ["north", "southeast"]
        else:
            return None

    # Return the directions of the sides that are adjacent to a specified vertex
    def get_adjacent_sides_of_vertex(self, direction):
        if direction == "northwest":
            return ["northwest", "west"]
        elif direction == "north":
            return ["northwest", "northeast"]
        elif direction == "northeast":
            return ["northeast", "east"]
        elif direction == "southeast":
            return ["southeast", "east"]
        elif direction == "south":
            return ["southeast", "southwest"]
        elif direction == "southwest":
            return ["southwest", "west"]
        else:
            return None

    # Check if a specified vertex is adjacent to a road
    def is_vertex_adjacent_to_road(self, direction):
        adjacent_sides = self.get_adjacent_sides_of_vertex(direction)
        for side in adjacent_sides:
            if self.get_side_from_direction(side) == 1:
                return True
        return False

    # Return a list of all the side values of this tile
    # Should start from east and go clockwise
    def get_all_side_values_as_list(self):
        return [
            self.side_east,
            self.side_southeast,
            self.side_southwest,
            self.side_west,
            self.side_northwest,
            self.side_northeast,
        ]

    # Return a list of all the vertex values of this tile
    def get_all_vertex_values_as_list(self):
        return [
            self.vert_northeast,
            self.vert_southeast,
            self.vert_south,
            self.vert_southwest,
            self.vert_northwest,
            self.vert_north,
        ]


# To do: Add type hints to getters and setters
