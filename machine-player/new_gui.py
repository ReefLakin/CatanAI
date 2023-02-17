# path: machine-player/new_gui.py

# Imports

# PYGAME
import pygame

# MATH
import math

# CATAN GAME
from CatanGame import CatanGame


# Set Constants

# TILE COLOURS
FIELD_TILE_COLOUR = "#f9c549"
MOUNTAIN_TILE_COLOUR = "#534e5a"
HILLS_TILE_COLOUR = "#be753b"
PASTURE_TILE_COLOUR = "#84b54b"
FOREST_TILE_COLOUR = "#3d4e26"
DESERT_TILE_COLOUR = "#bf9b61"
WATER_TILE_COLOUR = "#039cdd"

# OTHER COLOURS
BORDER_COLOUR = "#000000"
SETTLEMENT_COLOUR = "#FFFFFF"
ROAD_COLOUR = "#FFFFFF"
NUMBER_COLOUR = "#000000"
RED_NUMBER_COLOUR = "#BF4444"

# BOARD DIMENSIONS
BOARD_DIMS = [3, 4, 5, 4, 3]

# OTHER BOARD INFORMATION
STARTING_HEX_CENTRE_X = 250
STARTING_HEX_CENTRE_Y = 100
HEX_RADIUS = 60
SETTLEMENT_RADIUS = HEX_RADIUS / 6

# THE ACTUAL BOARD
game_instance = CatanGame()


# Helper Functions

# Collect the points for a single hexagon
def get_hex_points(centre_x, centre_y, radius, vertex_states, side_states):
    hex_points = []
    settle_points = []
    road_points = []
    # Make equiangular steps around the circle enclosing our hexagon
    for i in range(6):
        # Calculate the radian angle using complex maths
        a = (math.pi / 180) * (60 * i - 30)
        # Calculate the cartesian coordinates for a given angle and radius
        x = centre_x + radius * math.cos(a)
        y = centre_y + radius * math.sin(a)
        # Add the point to the list
        hex_points.append([x, y])

        # Also append x and y if the vertex is a settlement
        if vertex_states[i] == 1:
            settle_points.append([x, y])

        # This part is a little more complex
        # Take the current coordinates and the next coordinates if the current side is a road
        if side_states[i] == 1:
            # Calculate the radian angle using some fun maths
            a = (math.pi / 180) * (60 * (i + 1) - 30)
            # Calculate the cartesian coordinates for the given angle and radius
            x2 = centre_x + radius * math.cos(a)
            y2 = centre_y + radius * math.sin(a)
            # Add all the points to the list
            road_points.append([x, y, x2, y2])

    # Return the list of points
    return hex_points, settle_points, road_points


# Collect the points for each hexagon on the board
def get_all_hex_points(
    centre_x, centre_y, radius, board_dims, vertex_states, side_states
):
    x = centre_x
    y = centre_y
    vert_data = vertex_states
    side_data = side_states
    all_hex_points = []
    all_hex_centre_values = []
    all_settlement_points = []
    all_road_points = []

    # Loops through each row of the board according to the dimensions
    for row in range(len(board_dims)):
        # Recalculate the x-position based on the board dimensions
        hexesInRow = board_dims[row]
        e = hexesInRow - board_dims[0]
        x = centre_x - (e * (math.sqrt(3) * radius / 2))

        # Collect the points for each hexagon
        for hex in range(hexesInRow):
            # Get the centre points for the current hexagon
            all_hex_centre_values.append([x, y])
            # Pop the first element from the vertex data list
            current_hex_vert_data = vert_data.pop(0)
            # Pop the first element from the side data list
            current_hex_side_data = side_data.pop(0)
            # Get the outer points for the current hexagon
            hex_points, settle_points, road_points = get_hex_points(
                x, y, radius, current_hex_vert_data, current_hex_side_data
            )
            # Recalculate the x-position for the next hexagon
            x += math.sqrt(3) * radius

            # Add the hex points to the list
            all_hex_points.append(hex_points)
            # Add the settlement points to the list
            all_settlement_points.append(settle_points)
            # Add the road points to the list
            all_road_points.append(road_points)

        # Recalculate the y-position for the next row
        y += 3 / 4 * (radius * 2)

    # Return the list of points
    return all_hex_points, all_hex_centre_values, all_settlement_points, all_road_points


# Return the default type map associated with the default Catan board
def get_default_type_map():
    DEFAULT_TYPE_MAP = [1, 3, 4, 0, 2, 3, 2, 0, 4, 5, 4, 1, 4, 1, 0, 3, 2, 0, 3]
    return DEFAULT_TYPE_MAP


# Return the colour value associated with a given tile type id
def get_colour_value_from_id(id):
    match id:
        case 0:
            return FIELD_TILE_COLOUR
        case 1:
            return MOUNTAIN_TILE_COLOUR
        case 2:
            return HILLS_TILE_COLOUR
        case 3:
            return PASTURE_TILE_COLOUR
        case 4:
            return FOREST_TILE_COLOUR
        case 5:
            return DESERT_TILE_COLOUR
        case 6:
            return WATER_TILE_COLOUR


# Return the colour value associated with a given resource type name
def get_colour_value_from_resource_name(name):
    match name:
        case "grain":
            return FIELD_TILE_COLOUR
        case "ore":
            return MOUNTAIN_TILE_COLOUR
        case "brick":
            return HILLS_TILE_COLOUR
        case "wool":
            return PASTURE_TILE_COLOUR
        case "lumber":
            return FOREST_TILE_COLOUR
        case "desert":
            return DESERT_TILE_COLOUR
        case "water":
            return WATER_TILE_COLOUR


# Setup the Pygame Window

# Initialise Pygame
# Pygame's 'font' module is automatically initialised upon this call
pygame.init()

# Set the dimensions of the screen
screen = pygame.display.set_mode((900, 600))

# Fill the background colour to the screen
# Use pygame.Color to convert the hex colour to RGB
screen.fill(pygame.Color(WATER_TILE_COLOUR))

# Set the caption of the screen
pygame.display.set_caption("Catan Board")

# Load a font
font = pygame.font.SysFont("Arial", 18)


# Acquire Board Data

# Set the board to an in-progress test board
game_instance.setup_game_in_progress()

# Get the default type map
# type_map = get_default_type_map()

# Get the game state once
game_state = game_instance.get_state()

# Get the type map from the game instance
type_map = game_state["tile_types"]

# Get the values map from the game instance
values_map = game_state["tile_values"]

# Get the vertex states from the game instance
vertex_states = game_state["vertex_states"]

# Get the side states from the game instance
side_states = game_state["side_states"]

# Get the number of current victory points from the game instance
victory_points = game_state["victory_points"]

# Get the number of resources from the current player's resource pool
num_lumber = game_state["num_lumber"]
num_grain = game_state["num_grain"]
num_ore = game_state["num_ore"]
num_wool = game_state["num_wool"]
num_brick = game_state["num_brick"]


# Draw the Board

# Get the points for each hexagon on the board
(
    all_hex_points,
    all_hex_centre_values,
    all_settlement_points,
    all_road_points,
) = get_all_hex_points(
    STARTING_HEX_CENTRE_X,
    STARTING_HEX_CENTRE_Y,
    HEX_RADIUS,
    BOARD_DIMS,
    vertex_states,
    side_states,
)

# Draw each hexagon, with a border of 10 pixels
for hex_points in all_hex_points:
    # Get the colour value for the current hexagon
    fill_colour = get_colour_value_from_resource_name(type_map.pop(0))
    # Draw the hexagon
    pygame.draw.polygon(screen, pygame.Color(fill_colour), hex_points, 0)
    # Draw the border
    pygame.draw.polygon(screen, pygame.Color(BORDER_COLOUR), hex_points, 6)

    # Draw the value in the centre of each hexagon
    # Get the value for the current hexagon
    value = values_map.pop(0)
    # Get the centre points for the current hexagon
    centre_points = all_hex_centre_values.pop(0)
    # Skip the desert tile
    if value != 0:
        # Draw the text with font.render()
        # Is the number a six or an eight?
        if value == 6 or value == 8:
            text = font.render(str(value), True, pygame.Color(RED_NUMBER_COLOUR))
        else:
            text = font.render(str(value), True, pygame.Color(NUMBER_COLOUR))
        # Draw the text to the screen
        text_rect = text.get_rect(center=(centre_points[0], centre_points[1]))
        screen.blit(text, text_rect)

# Draw the settlements
# First, loop through each hex
for hex in all_settlement_points:
    # Then, loop through each vertex in the hex
    for vertex in hex:
        # Calculate x and y of the top left corner
        x = vertex[0] - SETTLEMENT_RADIUS
        y = vertex[1] - SETTLEMENT_RADIUS
        # Draw the settlement
        pygame.draw.rect(
            screen,
            pygame.Color(SETTLEMENT_COLOUR),
            (x, y, SETTLEMENT_RADIUS * 2, SETTLEMENT_RADIUS * 2),
            0,
        )

# Draw the roads
# First, loop through each hex
for hex in all_road_points:
    # Then, loop through each side in the hex
    for side in hex:
        # Draw the road
        pygame.draw.line(
            screen,
            pygame.Color(ROAD_COLOUR),
            (side[0], side[1]),
            (side[2], side[3]),
            6,
        )

# Write the victory points to the screen
# Draw the text with font.render()
text = font.render(
    "Victory Points: " + str(victory_points), True, pygame.Color(BORDER_COLOUR)
)
# Draw the text to the screen
text_rect = text.get_rect(center=(100, 570))
screen.blit(text, text_rect)

# Write the resources to the screen
# Draw the text with font.render()
lum_text = font.render("Lumber: " + str(num_lumber), True, pygame.Color(BORDER_COLOUR))
grain_text = font.render("Grain: " + str(num_grain), True, pygame.Color(BORDER_COLOUR))
ore_text = font.render("Ore: " + str(num_ore), True, pygame.Color(BORDER_COLOUR))
wool_text = font.render("Wool: " + str(num_wool), True, pygame.Color(BORDER_COLOUR))
brick_text = font.render("Brick: " + str(num_brick), True, pygame.Color(BORDER_COLOUR))
# Draw the text to the screen
lumber_text_rect = lum_text.get_rect(center=(800, 30))
grain_text_rect = grain_text.get_rect(center=(800, 50))
ore_text_rect = ore_text.get_rect(center=(800, 70))
wool_text_rect = wool_text.get_rect(center=(800, 90))
brick_text_rect = brick_text.get_rect(center=(800, 110))
screen.blit(lum_text, lumber_text_rect)
screen.blit(grain_text, grain_text_rect)
screen.blit(ore_text, ore_text_rect)
screen.blit(wool_text, wool_text_rect)
screen.blit(brick_text, brick_text_rect)


# Update the display using flip
pygame.display.flip()


# Game Loop

# Variable to keep our game loop running
running = True

while running:

    # Loop over the event queue
    for event in pygame.event.get():

        # Check for QUIT event
        if event.type == pygame.QUIT:
            running = False
