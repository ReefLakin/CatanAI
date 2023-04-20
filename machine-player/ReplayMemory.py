# Imports
import random
import json
import os
from datetime import datetime


class ReplayMemory:
    def __init__(self, buffer_max_size):
        self.buffer = []
        self.buffer_max_size = buffer_max_size

    # Add an experience to the buffer
    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    # Sample a batch of experiences from the buffer
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    # Get the buffer
    def get_buffer(self):
        return self.buffer

    # Get the current size of the buffer
    def get_buffer_size(self):
        return len(self.buffer)

    # Save the buffer to a file
    def save_buffer(self, agent_name="Agent"):
        # Convert list of tuples to JSON
        json_data = json.dumps(self.buffer)

        # Create directory for history files if it doesn't exist
        directory = "history"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Get current date and time for file name
        now = datetime.now()
        filename = agent_name + now.strftime("-%Y-%m-%d-%H-%M") + ".json"

        # Write JSON data to file
        with open(os.path.join(directory, filename), "w") as file:
            file.write(json_data)
