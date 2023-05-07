# path: machine-player/headlesser_training.py

# Imports
from TrainingSession import TrainingSession


# !! Main Program

# # Training Session Options
agent_to_set = "Adam"
opponents_to_set = ["Randy", "Adam", "Randy"]
player_count = len(opponents_to_set) + 1

# Create a training session (with default parameters)
training_session = TrainingSession(agent=agent_to_set, opponents=opponents_to_set)


# # Game Loop

# Start the training session
running = training_session.start(players=player_count)

# While the training session is running
while running is True:
    (
        running,
        legal_actions,
        chosen_action,
        games_played,
        current_player,
        other_information,
    ) = training_session.time_step()
    print(f"Game Number: {games_played} / Recent Action: {chosen_action}")
