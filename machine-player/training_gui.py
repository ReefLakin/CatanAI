# path: machine-player/newer_gui.py

# Imports
from TrainingSession import TrainingSession
import pygame
import time
import math
import numpy as np
import dotenv

# Load necessary variables from the .env file
human_player = dotenv.get_key(".env", "HUMAN_PLAYER")

# Constants specific to the GUI version of the program
# TILE COLOURS
FIELD_TILE_COLOUR = "#f9c549"
MOUNTAIN_TILE_COLOUR = "#534e5a"
HILLS_TILE_COLOUR = "#be753b"
PASTURE_TILE_COLOUR = "#84b54b"
FOREST_TILE_COLOUR = "#3d4e26"
DESERT_TILE_COLOUR = "#bf9b61"
WATER_TILE_COLOUR = "#039cdd"

other_information = {"agent_index": 0}

# OTHER COLOURS
BORDER_COLOUR = "#000000"
SETTLEMENT_COLOUR = "#FFFFFF"
ROAD_COLOUR = "#FFFFFF"
NUMBER_COLOUR = "#000000"
RED_NUMBER_COLOUR = "#BF4444"
VP_COLOUR = "#1B152E"
CITY_GOLD_COLOUR = "#FFD700"
ROBBER_COLOUR = "#C4BCA9"

# PLAYER COLOURS
PLAYER_0_COLOUR = "#FFFFFF"  # White
PLAYER_1_COLOUR = "#FFBB00"  # Orange
PLAYER_2_COLOUR = "#FF0000"  # Red
PLAYER_3_COLOUR = "#0000FF"  # Blue

# SCREEN INFORMATION
SCREEN_WIDTH = 880
SCREEN_HEIGHT = 700

# OTHER BOARD INFORMATION
STARTING_HEX_CENTRE_X = 330
STARTING_HEX_CENTRE_Y = 140
HEX_RADIUS = 60
SETTLEMENT_RADIUS = HEX_RADIUS / 7
CITY_RADIUS = HEX_RADIUS / 8


# Helper Functions
# Collect the points for a single hexagon
def get_hex_points(
    centre_x, centre_y, radius, vertex_states, vertex_owners, side_states, side_owners
):
    hex_points = []
    settle_points = []
    settle_owners = []
    city_points = []
    city_owners = []
    road_points = []
    road_owners = []
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
            # Add to the list of settlement owners
            settle_owners.append(vertex_owners[i])

        # Also append x and y if the vertex is a city
        if vertex_states[i] == 2:
            city_points.append([x, y])
            # Add to the list of city owners
            city_owners.append(vertex_owners[i])

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
            # Add to the list of road owners
            road_owners.append(side_owners[i])

    # Return the list of points
    return (
        hex_points,
        settle_points,
        settle_owners,
        road_points,
        road_owners,
        city_points,
        city_owners,
    )


# Collect the points for each hexagon on the board
def get_all_hex_points(
    centre_x,
    centre_y,
    radius,
    board_dims,
    vertex_states,
    vertex_owners,
    side_states,
    side_owners,
    robber_states,
):
    x = centre_x
    y = centre_y
    vert_data = vertex_states
    side_data = side_states
    all_hex_points = []
    all_hex_centre_values = []
    all_settlement_points = []
    all_settlement_owners = []
    all_road_points = []
    all_road_owners = []
    all_city_points = []
    all_city_owners = []
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
            # Pop the next element from the vertex data list
            current_hex_vert_data = vert_data.pop(0)
            # Pop the next element from the side data list
            current_hex_side_data = side_data.pop(0)
            # Pop the next element from the vertex and side owner lists
            current_hex_vert_owners = vertex_owners.pop(0)
            current_hex_side_owners = side_owners.pop(0)
            # Get the outer points for the current hexagon
            (
                hex_points,
                settle_points,
                settle_owners,
                road_points,
                road_owners,
                city_points,
                city_owners,
            ) = get_hex_points(
                x,
                y,
                radius,
                current_hex_vert_data,
                current_hex_vert_owners,
                current_hex_side_data,
                current_hex_side_owners,
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
            # Add the settlement owners to the list
            all_settlement_owners.append(settle_owners)
            # Add the road owners to the list
            all_road_owners.append(road_owners)
            # Add the city owners to the list
            all_city_owners.append(city_owners)

            # Increase the robber_index by 1
            robber_index += 1

        # Recalculate the y-position for the next row
        y += 3 / 4 * (radius * 2)

    # Return the list of points
    return (
        all_hex_points,
        all_hex_centre_values,
        all_settlement_points,
        all_settlement_owners,
        all_road_points,
        all_road_owners,
        all_city_points,
        all_city_owners,
        robber_points,
    )


# Return the player colour associated with a given player index
def get_colour_name_from_index(index):
    match index:
        case 0:
            return "White"
        case 1:
            return "Orange"
        case 2:
            return "Red"
        case 3:
            return "Blue"


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


# Return the player colour associated with the given player ID
def get_player_colour_from_id(id):
    match id:
        case 0:
            return PLAYER_0_COLOUR
        case 1:
            return PLAYER_1_COLOUR
        case 2:
            return PLAYER_2_COLOUR
        case 3:
            return PLAYER_3_COLOUR


# !! Main Program

# # Training Session Options
agent_to_set = "Adam"
opponents_to_set = ["Johnny", "James", "Ethan"]
player_count = len(opponents_to_set) + 1
use_pixel_data_instead_of_state = False

# Create a training session (with default parameters)
training_session = TrainingSession(
    agent=agent_to_set,
    opponents=opponents_to_set,
    use_pixels=use_pixel_data_instead_of_state,
)


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
opp_res_font = pygame.font.SysFont("Arial", 14)
sm_font = pygame.font.SysFont("Arial", 10)
vp_font = pygame.font.SysFont("Arial", 20, bold=True)

# Set the current player
current_player = 0


# # Game Loop

# Start the training session
running = training_session.start(players=player_count)
fast_training_on = False
sleep_time = 0.03

# While the training session is running
while running is True:

    # If quick training is on, then take an action and update the board
    if fast_training_on:
        time.sleep(sleep_time)
        pygame.event.post(take_action)
        pygame.event.post(update_game_board)

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

            # Check for 1 key press (SPEED 1)
            if event.key == pygame.K_1:
                sleep_time = 0.5

            # Check for 2 key press (SPEED 2)
            if event.key == pygame.K_2:
                sleep_time = 0.1

            # Check for 3 key press (SPEED 3)
            if event.key == pygame.K_3:
                sleep_time = 0.05

            # Check for 4 key press (SPEED 4)
            if event.key == pygame.K_4:
                sleep_time = 0.03

            # Check for 5 key press (SPEED 5)
            if event.key == pygame.K_5:
                sleep_time = 0.01

            # Check for 6 key press (SPEED 6)
            if event.key == pygame.K_6:
                sleep_time = 0.005

            # Check for 7 key press (SPEED 7)
            if event.key == pygame.K_7:
                sleep_time = 0.001

            # Check for 8 key press (SPEED 8)
            if event.key == pygame.K_8:
                sleep_time = 0.0005

            # Check for 9 key press (SPEED 9)
            if event.key == pygame.K_9:
                sleep_time = 0.0001

            # Check for 0 key press (SPEED 0)
            if event.key == pygame.K_0:
                sleep_time = 0.00001

            # Check for T key press (TRAIN)
            if event.key == pygame.K_t:
                # Set the flag indicating fast training if it's not already on
                if not fast_training_on:
                    fast_training_on = True
                # Otherwise, turn it off
                else:
                    fast_training_on = False

                # Take action, then update the board
                pygame.event.post(take_action)
                pygame.event.post(update_game_board)

        # Check for the custom event that we set up to take action
        if event.type == TAKE_ACTION:
            (
                running,
                legal_actions,
                chosen_action,
                games_played,
                current_player,
                other_information,
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
            vertex_owners = game_state["vertex_owners"]

            # Get the side states from the game instance
            side_states = game_state["side_states"]
            side_owners = game_state["side_owners"]

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

            # Get the game state for each opponent
            opp_game_state_1 = game_instance.get_state(player_id=1)
            opp_game_state_2 = game_instance.get_state(player_id=2)
            opp_game_state_3 = game_instance.get_state(player_id=3)

            # Draw the Board

            # Get the points for each hexagon on the board
            (
                all_hex_points,
                all_hex_centre_values,
                all_settlement_points,
                all_settlement_owners,
                all_road_points,
                all_road_owners,
                all_city_points,
                all_city_owners,
                robber_points,
            ) = get_all_hex_points(
                STARTING_HEX_CENTRE_X,
                STARTING_HEX_CENTRE_Y,
                HEX_RADIUS,
                training_session.get_board_dims(),
                vertex_states,
                vertex_owners,
                side_states,
                side_owners,
                robber_states,
            )

            # Completely flatten each of the ownership arrays
            all_settlement_owners = [
                item for sublist in all_settlement_owners for item in sublist
            ]
            all_road_owners = [item for sublist in all_road_owners for item in sublist]
            all_city_owners = [item for sublist in all_city_owners for item in sublist]

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
                        pygame.Color(
                            get_player_colour_from_id(all_settlement_owners.pop(0))
                        ),
                        (x, y, SETTLEMENT_RADIUS * 2, SETTLEMENT_RADIUS * 2),
                        0,
                    )

            # Draw the cities
            # First, loop through each hex
            for hex in all_city_points:
                # Then, loop through each vertex in the hex
                for vertex in hex:
                    city_col = get_player_colour_from_id(all_city_owners.pop(0))
                    city_diameter = CITY_RADIUS * 2
                    # Calculate x and y of the top left corner (inner white square)
                    x = vertex[0] - CITY_RADIUS
                    y = vertex[1] - CITY_RADIUS * 2
                    # Draw the city (outer gold square)
                    pygame.draw.rect(
                        screen,
                        pygame.Color(city_col),
                        (x, y, city_diameter, city_diameter * 1.6),
                        0,
                    )

                    # Calculate x and y of the top left corner (inner white square)
                    x = vertex[0] - CITY_RADIUS
                    y = vertex[1]
                    # Draw the city (outer gold square)
                    pygame.draw.rect(
                        screen,
                        pygame.Color(city_col),
                        (x, y, city_diameter * 1.8, city_diameter),
                        0,
                    )

                    # draw the triangle on top
                    x = vertex[0] - CITY_RADIUS
                    y = vertex[1] - CITY_RADIUS * 2
                    triangle_points = [
                        (x, y),
                        (x + city_diameter, y),
                        (x + CITY_RADIUS, y - 10),
                    ]
                    pygame.draw.polygon(screen, city_col, triangle_points)

            # Draw the roads
            # First, loop through each hex
            for hex in all_road_points:
                # Then, loop through each side in the hex
                for side in hex:
                    # Draw the road
                    pygame.draw.line(
                        screen,
                        pygame.Color(get_player_colour_from_id(all_road_owners.pop(0))),
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

            # Write the opponent's victory points to the screen
            # Draw the text with font.render()
            text = vp_font.render(
                " / " + str(opp_game_state_1["victory_points"]),
                True,
                pygame.Color(PLAYER_1_COLOUR),
            )
            # Draw the text to the screen
            text_rect = text.get_rect(center=(185, 25))
            screen.blit(text, text_rect)

            # If there are more than 2 players, write the third player's victory points to the screen
            if player_count > 2:
                # Draw the text with font.render()
                text = vp_font.render(
                    " / " + str(opp_game_state_2["victory_points"]),
                    True,
                    pygame.Color(PLAYER_2_COLOUR),
                )
                # Draw the text to the screen
                text_rect = text.get_rect(center=(215, 25))
                screen.blit(text, text_rect)

            # If there are more than 3 players, write the fourth player's victory points to the screen
            if player_count > 3:
                # Draw the text with font.render()
                text = vp_font.render(
                    " / " + str(opp_game_state_3["victory_points"]),
                    True,
                    pygame.Color(PLAYER_3_COLOUR),
                )
                # Draw the text to the screen
                text_rect = text.get_rect(center=(245, 25))
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

            # Write who is currently playing to the screen in the top right corner
            text = font.render(
                "Current Player: " + str(current_player),
                True,
                pygame.Color(VP_COLOUR),
            )

            # Draw the text to the screen
            text_rect = text.get_rect(center=(SCREEN_WIDTH - 80, 50))
            screen.blit(text, text_rect)

            # Write up the resource texts
            resource_texts = [
                (f"Lumber: {num_lumber}"),
                (f"Grain: {num_grain}"),
                (f"Ore: {num_ore}"),
                (f"Wool: {num_wool}"),
                (f"Brick: {num_brick}"),
            ]

            # List the opponent resource texts
            opponent_resources_1 = [
                str(opp_game_state_1["num_lumber"]),
                str(opp_game_state_1["num_grain"]),
                str(opp_game_state_1["num_ore"]),
                str(opp_game_state_1["num_wool"]),
                str(opp_game_state_1["num_brick"]),
            ]

            # If there are more than 2 players, list the second opponent resource texts
            if player_count > 2:
                opponent_resources_2 = [
                    str(opp_game_state_2["num_lumber"]),
                    str(opp_game_state_2["num_grain"]),
                    str(opp_game_state_2["num_ore"]),
                    str(opp_game_state_2["num_wool"]),
                    str(opp_game_state_2["num_brick"]),
                ]

            # If there are more than 3 players, list the third opponent resource texts
            if player_count > 3:
                opponent_resources_3 = [
                    str(opp_game_state_3["num_lumber"]),
                    str(opp_game_state_3["num_grain"]),
                    str(opp_game_state_3["num_ore"]),
                    str(opp_game_state_3["num_wool"]),
                    str(opp_game_state_3["num_brick"]),
                ]

            # Load each of the resource icons
            lumber_icon = pygame.image.load("assets/wood.png").convert_alpha()
            brick_icon = pygame.image.load("assets/wall.png").convert_alpha()
            wool_icon = pygame.image.load("assets/sheep.png").convert_alpha()
            grain_icon = pygame.image.load("assets/wheat.png").convert_alpha()
            ore_icon = pygame.image.load("assets/stone.png").convert_alpha()

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

            # Loop through the opponent's resource texts
            for i in range(len(opponent_resources_1)):
                # Create resource text
                opp_res_txt = (
                    opponent_resources_1[i]
                    + " / "
                    + opponent_resources_2[i]
                    + " / "
                    + opponent_resources_3[i]
                )
                # Create the surface
                text_surface = opp_res_font.render(opp_res_txt, True, "#675df2")
                text_rect = text_surface.get_rect()
                # Set the x position
                text_rect.x = box_rect.x + text_spacing + (font_size + text_spacing) * i
                # Set the y position
                text_rect.y = box_y + (box_height - font_size) // 2 + 20
                # Add the surface and rect to the list
                text_surfaces.append((text_surface, text_rect))

            # Blit the text surfaces onto the window
            for text_surface, text_rect in text_surfaces:
                screen.blit(text_surface, text_rect)

            # Write the agent's name to the screen in the bottom left corner
            agent_index = other_information["agent_index"]
            text_to_display = (
                f"{agent_to_set} ({get_colour_name_from_index(agent_index)})"
            )
            text = sm_font.render(text_to_display, True, pygame.Color(BORDER_COLOUR))
            # Draw the text to the screen
            text_rect = text.get_rect(center=(45, SCREEN_HEIGHT - 15))
            screen.blit(text, text_rect)

            # Update the display using flip
            pygame.display.flip()

            if use_pixel_data_instead_of_state:
                # Collect the pixel data
                pixel_data = pygame.surfarray.array3d(pygame.display.get_surface())

                # Feed pixel data to the training session object
                training_session.feed_pixel_data(pixel_data)
