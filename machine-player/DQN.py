# DQN class for the machine player

# Imports
import torch.nn as nn
import torch


class DQN(nn.Module):
    def __init__(self, input_size=272, output_size=382, hidden_size=50):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.LeakyReLU()
        self.drop3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.LeakyReLU()
        self.drop4 = nn.Dropout(p=0.2)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.relu5 = nn.LeakyReLU()
        self.drop5 = nn.Dropout(p=0.2)
        self.fc6 = nn.Linear(hidden_size, output_size)

        # Initialise weights with the He uniform method
        # Loop through all layers and initialise the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # a represents the slope of the negative part of the rectifier, default is 0
                # mode is the method of initialisation
                # nonlinearity is the activation function
                nn.init.kaiming_uniform_(
                    m.weight, a=0, mode="fan_in", nonlinearity="relu"
                )
                # Initialise biases to zero, which is the default anyway
                nn.init.zeros_(m.bias)

        # Initialise the optimiser
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.002, amsgrad=True)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.drop4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.drop5(x)
        x = self.fc6(x)
        return x
