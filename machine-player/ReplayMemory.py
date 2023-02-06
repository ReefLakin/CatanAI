import random

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        
    # Add an experience to the buffer
    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
        
    # Sample a batch of experiences from the buffer
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
