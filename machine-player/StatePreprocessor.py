# Class for preprocessing the state of the game
class StatePreprocessor:

    # Preprocess the state information passed to the select_action method
    def preprocess_state(self, state):
        new_state_list = []

        # Victory points
        new_state_list.append(state["victory_points"])

        # Turn number
        new_state_list.append(state["turn_number"])

        # Total number of ore
        new_state_list.append(state["num_ore"])

        # Total number of grain
        new_state_list.append(state["num_grain"])

        # Total number of wool
        new_state_list.append(state["num_wool"])

        # Total number of lumber
        new_state_list.append(state["num_lumber"])

        # Total number of brick
        new_state_list.append(state["num_brick"])

        # Loop over each tile for each side
        for tile in state["side_states"]:
            # Loop over each side
            for side in tile:
                # Add the side to the new state list
                new_state_list.append(side)

        # Loop over each tile for each vertex
        for tile in state["vertex_states"]:
            # Loop over each vertex
            for vertex in tile:
                # Add the vertex to the new state list
                new_state_list.append(vertex)

        # Loop over tile types
        for tile_type in state["tile_types"]:
            # Change string values to integers
            # brick = 1, lumber = 2, wool = 3, grain = 4, ore = 5, desert = 6
            if tile_type == "brick":
                new_state_list.append(1)
            elif tile_type == "lumber":
                new_state_list.append(2)
            elif tile_type == "wool":
                new_state_list.append(3)
            elif tile_type == "grain":
                new_state_list.append(4)
            elif tile_type == "ore":
                new_state_list.append(5)
            elif tile_type == "desert":
                new_state_list.append(6)

        # Loop over tile numbers
        for tile_number in state["tile_values"]:
            # Add the tile number to the new state list
            new_state_list.append(tile_number)

        # Loop over the board dimensions
        for row in state["board_dims"]:
            # Add the board dimension to the new state list
            new_state_list.append(row)

        # Return the new state list where None values are replaced with 0 (through list comprehension)
        return [0 if v is None else v for v in new_state_list]
