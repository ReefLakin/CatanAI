class Road:
    def __init__(self, owner=1):
        self.owner = owner

    # A road is equal to an integer if the integer is 1
    def __eq__(self, other):
        if isinstance(other, int):
            return 1 == other
        return False

    def get_owner(self):
        return self.owner

    def set_owner(self, owner):
        self.owner = owner
