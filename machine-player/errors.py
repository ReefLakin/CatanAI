# Python file for writing custom error messages


class AgentCompatibilityError(Exception):
    # Raised when the agent is not compatible with the environment
    # Likely if the training env is set up for pixel input, but state is used (or vice versa)
    def __init__(self, message):
        self.message = message
