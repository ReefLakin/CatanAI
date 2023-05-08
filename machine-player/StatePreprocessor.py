# This class is used to preprocess the state information passed to the select_action method.
# It takes the complex lists of state information and converts them into a list of integers.


# Class definition
class StatePreprocessor:
    def preprocess_state(self, state):
        # Create a new list to hold the new state information
        new_state_list = []

        # Victory points
        new_state_list.append(state["victory_points"])

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

        # For each hex edge on the board, if it's 0, it's not owned. If it's 1, it's owned by player 1, etc.
        for tile in state["side_owners"]:
            for side in tile:
                if side == None:
                    new_state_list.append(None)
                else:
                    new_state_list.append(side + 1)

        # For each hex vertex on the board, if it's 0, it's not owned. If it's 1, it's owned by player 1, etc.
        for tile in state["vertex_owners"]:
            for vertex in tile:
                if vertex == None:
                    new_state_list.append(None)
                else:
                    new_state_list.append(vertex + 1)

        # For each tile on the board, add its type to the list in numerical form
        # brick = 1, lumber = 2, wool = 3, grain = 4, ore = 5, desert = 0
        for tile_type in state["tile_types"]:
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
                new_state_list.append(0)

        # For each tile on the board, add its production value to the list
        # The desert and tile containing the robber have a production value of 0
        tile_vals = state["tile_values"]
        robber_vals = state["robber_states"]
        for i in range(len(tile_vals)):
            if robber_vals[i] == 1:
                new_state_list.append(0)
            else:
                new_state_list.append(tile_vals[i])

        # Return the new state list
        # None-type values are replaced with 0 (through list comprehension)
        return [0 if v is None else v for v in new_state_list]
