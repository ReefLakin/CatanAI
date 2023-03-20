class Township:
    def __init__(self, owner, type):
        self.owner = owner
        self.type = type  # 1 (settlement) or 2 (city)

    def get_owner(self):
        return self.owner

    def get_type(self):
        return self.type

    def set_owner(self, owner):
        self.owner = owner

    def set_type(self, type):
        self.type = type

    def upgrade(self):
        self.type = 2
