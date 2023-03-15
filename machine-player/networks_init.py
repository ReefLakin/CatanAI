import torch
import QNetwork

# Init. the Q-network
qnetwork = QNetwork()

# Init. the target network
target_network = QNetwork()

# Copy the parameters from the Q-network to the target network
target_network.load_state_dict(qnetwork.state_dict())

# Move the networks to the GPU (if possible)
if torch.cuda.is_available():
    qnetwork.cuda()
    target_network.cuda()

# Set target network to evaluation mode
target_network.eval()
