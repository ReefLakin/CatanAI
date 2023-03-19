# path: machine-player/newer_gui.py

# Imports
from TrainingSession import TrainingSession
import pygame
import time
import math

# Constants specific to the GUI version of the program
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
VP_COLOUR = "#1B152E"
CITY_GOLD_COLOUR = "#FFD700"
ROBBER_COLOUR = "#C4BCA9"

# SCREEN INFORMATION
SCREEN_WIDTH = 880
SCREEN_HEIGHT = 700

# OTHER BOARD INFORMATION
STARTING_HEX_CENTRE_X = 330
STARTING_HEX_CENTRE_Y = 140
HEX_RADIUS = 60
SETTLEMENT_RADIUS = HEX_RADIUS / 7
CITY_RADIUS_OUTER = HEX_RADIUS / 5
CITY_RADIUS_INNER = HEX_RADIUS / 6


# Helper Functions
# Collect the points for a single hexagon
def get_hex_points(centre_x, centre_y, radius, vertex_states, side_states):
    hex_points = []
    settle_points = []
    city_points = []
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

        # Also append x and y if the vertex is a city
        if vertex_states[i] == 2:
            city_points.append([x, y])

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
    return hex_points, settle_points, road_points, city_points


# Collect the points for each hexagon on the board
def get_all_hex_points(
    centre_x, centre_y, radius, board_dims, vertex_states, side_states, robber_states
):
    x = centre_x
    y = centre_y
    vert_data = vertex_states
    side_data = side_states
    all_hex_points = []
    all_hex_centre_values = []
    all_settlement_points = []
    all_road_points = []
    all_city_points = []
    robber_points = []
    robber_index = 0

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
            # Does the robber occupy the current hexagon?
            if robber_states[robber_index] == 1:
                # Add the robber to the list
                robber_points.append([x, y])
            # Pop the first element from the vertex data list
            current_hex_vert_data = vert_data.pop(0)
            # Pop the first element from the side data list
            current_hex_side_data = side_data.pop(0)
            # Get the outer points for the current hexagon
            hex_points, settle_points, road_points, city_points = get_hex_points(
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
            # Add the city points to the list
            all_city_points.append(city_points)

            # Increase the robber_index by 1
            robber_index += 1

        # Recalculate the y-position for the next row
        y += 3 / 4 * (radius * 2)

    # Return the list of points
    return (
        all_hex_points,
        all_hex_centre_values,
        all_settlement_points,
        all_road_points,
        all_city_points,
        robber_points,
    )


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


# !! Main Program

# # Training Session Options
agent_to_set = "Randy"

# Create a training session (with default parameters)
training_session = TrainingSession(agent=agent_to_set)


# # Pygame Window Setup

# Call pygame.init() to initialise pygame
pygame.init()

# Custom events
UPDATE_GAME_BOARD_EVENT = pygame.USEREVENT + 0
TAKE_ACTION = pygame.USEREVENT + 1

# Create the event as a pygame event
update_game_board = pygame.event.Event(UPDATE_GAME_BOARD_EVENT)
take_action = pygame.event.Event(TAKE_ACTION)

# Post the event immediately
pygame.event.post(update_game_board)

# Set the dimensions of the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Fill the background colour to the screen
# Use pygame.Color to convert the hex colour to RGB
screen.fill(pygame.Color(WATER_TILE_COLOUR))

# Set the caption of the screen
pygame.display.set_caption("Catan Board")

# Load a font
font = pygame.font.SysFont("Arial", 18)
res_font = pygame.font.SysFont("Arial", 17)
sm_font = pygame.font.SysFont("Arial", 10)
vp_font = pygame.font.SysFont("Arial", 20, bold=True)


# # Game Loop

# Start the training session
running = training_session.start()
PLEASE_LORD_GIVE_ME_A_BREAK = False

# While the training session is running
while running is True:

    # If PLEASE_LORD_GIVE_ME_A_BREAK is true, make a small time delay
    if PLEASE_LORD_GIVE_ME_A_BREAK:
        time.sleep(0.004)
        PLEASE_LORD_GIVE_ME_A_BREAK = False
        pygame.event.post(take_action)
        pygame.event.post(update_game_board)
        PLEASE_LORD_GIVE_ME_A_BREAK = True

    # Loop over the event queue
    for event in pygame.event.get():

        # Check for QUIT event
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            # Check for S key press (STEP)
            if event.key == pygame.K_s:
                # Take action, then update the board
                pygame.event.post(take_action)
                pygame.event.post(update_game_board)

            # Check for T key press (TRAIN)
            if event.key == pygame.K_t:
                # Take action, then update the board
                pygame.event.post(take_action)
                pygame.event.post(update_game_board)
                PLEASE_LORD_GIVE_ME_A_BREAK = True

        # Check for the custom event that we set up to take action
        if event.type == TAKE_ACTION:

            (
                running,
                legal_actions,
                chosen_action,
                games_played,
            ) = training_session.time_step()

        # Check for the custom event that we set up to update the game board
        if event.type == UPDATE_GAME_BOARD_EVENT:

            # Update the game state on the GUI
            screen.fill(pygame.Color(WATER_TILE_COLOUR))

            # Get the game instance
            game_instance = training_session.get_game_instance()

            # Get the game state once
            game_state = training_session.get_game_state()

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

            # Get the robber states
            robber_states = game_state["robber_states"]

            # Draw the Board

            # Get the points for each hexagon on the board
            (
                all_hex_points,
                all_hex_centre_values,
                all_settlement_points,
                all_road_points,
                all_city_points,
                robber_points,
            ) = get_all_hex_points(
                STARTING_HEX_CENTRE_X,
                STARTING_HEX_CENTRE_Y,
                HEX_RADIUS,
                training_session.get_board_dims(),
                vertex_states,
                side_states,
                robber_states,
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
                        text = font.render(
                            str(value), True, pygame.Color(RED_NUMBER_COLOUR)
                        )
                    else:
                        text = font.render(
                            str(value), True, pygame.Color(NUMBER_COLOUR)
                        )
                    # Draw the text to the screen
                    text_rect = text.get_rect(
                        center=(centre_points[0], centre_points[1])
                    )
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

            # Draw the cities
            # First, loop through each hex
            for hex in all_city_points:
                # Then, loop through each vertex in the hex
                for vertex in hex:
                    # Calculate x and y of the top left corner (outer gold square)
                    x = vertex[0] - CITY_RADIUS_OUTER
                    y = vertex[1] - CITY_RADIUS_OUTER
                    # Draw the city (outer gold square)
                    pygame.draw.rect(
                        screen,
                        pygame.Color(CITY_GOLD_COLOUR),
                        (x, y, CITY_RADIUS_OUTER * 2, CITY_RADIUS_OUTER * 2),
                        0,
                    )
                    # Calculate x and y of the top left corner (inner white square)
                    x = vertex[0] - CITY_RADIUS_INNER
                    y = vertex[1] - CITY_RADIUS_INNER
                    # Draw the city (outer gold square)
                    pygame.draw.rect(
                        screen,
                        pygame.Color(SETTLEMENT_COLOUR),
                        (x, y, CITY_RADIUS_INNER * 2, CITY_RADIUS_INNER * 2),
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

            # Draw the robber
            # He'll just be a grey circle like the normal robber
            for robber in robber_points:
                pygame.draw.circle(screen, pygame.Color(ROBBER_COLOUR), robber, 15, 0)

            # Write the victory points to the screen
            # Draw the text with font.render()
            text = vp_font.render(
                "Victory Points: " + str(victory_points),
                True,
                pygame.Color(VP_COLOUR),
            )
            # Draw the text to the screen
            text_rect = text.get_rect(center=(90, 25))
            screen.blit(text, text_rect)

            # Write the turn number just below the victory points
            text = font.render(
                "Turn: " + str(game_instance.get_turn_number()),
                True,
                pygame.Color(VP_COLOUR),
            )
            # Draw the text to the screen
            text_rect = text.get_rect(center=(90, 50))
            screen.blit(text, text_rect)

            # Get the most recent dice roll
            most_recent_dice_roll = game_instance.get_most_recent_roll()
            roll_value = most_recent_dice_roll[2]

            # Write the dice roll to the screen in the top right corner
            text = font.render(
                "Last Dice Roll: " + str(roll_value),
                True,
                pygame.Color(VP_COLOUR),
            )
            # Draw the text to the screen
            text_rect = text.get_rect(center=(SCREEN_WIDTH - 80, 25))
            screen.blit(text, text_rect)

            # Write up the resource texts
            resource_texts = [
                (f"Lumber: {num_lumber}"),
                (f"Grain: {num_grain}"),
                (f"Ore: {num_ore}"),
                (f"Wool: {num_wool}"),
                (f"Brick: {num_brick}"),
            ]

            # Load each of the resource icons
            lumber_icon = pygame.image.load("wood.png").convert_alpha()
            brick_icon = pygame.image.load("wall.png").convert_alpha()
            wool_icon = pygame.image.load("sheep.png").convert_alpha()
            grain_icon = pygame.image.load("wheat.png").convert_alpha()
            ore_icon = pygame.image.load("stone.png").convert_alpha()

            # List the icons
            icons = [lumber_icon, grain_icon, ore_icon, wool_icon, brick_icon]

            # Set up the resource box at the bottom of the screen
            box_height = 100
            box_y = SCREEN_HEIGHT - box_height
            box_color = (255, 255, 255)
            box_border_color = (0, 0, 0)
            box_border_width = 2
            box_rect = pygame.Rect(0, box_y, SCREEN_WIDTH, box_height)

            # Draw the box
            pygame.draw.rect(screen, box_color, box_rect)
            pygame.draw.rect(screen, box_border_color, box_rect, box_border_width)

            # Work out the spacing between the resource texts
            font_size = 17
            text_spacing = (box_rect.width - font_size * 5) / 6
            text_surfaces = []

            # Get the spacing between the icons
            icon_spacing = (
                box_rect.width - (font_size + icons[0].get_width()) * 5
            ) // 6

            # Loop through the resource texts
            for i in range(len(resource_texts)):
                # Create the surface
                text_surface = res_font.render(resource_texts[i], True, (0, 0, 0))
                text_rect = text_surface.get_rect()
                # Set the x position
                text_rect.x = box_rect.x + text_spacing + (font_size + text_spacing) * i
                # Set the y position
                text_rect.y = box_y + (box_height - font_size) // 2
                # Add the surface and rect to the list
                text_surfaces.append((text_surface, text_rect))

                icon_surface = icons[i]
                if icon_surface:
                    icon_rect = icon_surface.get_rect()
                    icon_rect.x = text_rect.x - 30
                    icon_rect.y = text_rect.y
                    screen.blit(icon_surface, icon_rect)

            # Blit the text surfaces onto the window
            for text_surface, text_rect in text_surfaces:
                screen.blit(text_surface, text_rect)

            # Write the agent's name to the screen in the bottom left corner
            text = sm_font.render(agent_to_set, True, pygame.Color(BORDER_COLOUR))
            # Draw the text to the screen
            text_rect = text.get_rect(center=(25, SCREEN_HEIGHT - 15))
            screen.blit(text, text_rect)

            # Update the display using flip
            pygame.display.flip()
