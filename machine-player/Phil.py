# Imports
import numpy as np
import DeepQNetwork as DeepQNetwork
import torch as T

# Define the Phil class
class Phil:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        batch_size,
        n_action,
        max_mem_size=100000,
        eps_end=0.01,
        eps_dec=5e-4,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_action
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(self.n_actions)]
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork()
        self.state_memory = np.zeros(
            (self.mem_size, *self.input_dims), dtype=np.float32
        )
        self.new_state_memory = np.zeros(
            (self.mem_size, *self.input_dims), dtype=np.float32
        )
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def select_action(self, observation, all_possible_actions, legal_actions):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return all_possible_actions[action]

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )

    # Reward function
    def reward(self, reward_information):
        # Phil gets rewarded more for: reaching 10 VPs
        # Phil gets moderately rewarded for: building roads, settlements, cities
        # Phil gets mildly punished for: making illegal moves
        legal_actions = reward_information["legal_actions"]
        done = reward_information["game_over"]
        action = reward_information["current_action"]

        if done == True:
            return 80  # Victory reward
        elif action not in legal_actions:
            return -1  # Illegal move punishment
        else:
            split_action = action.split("_")
            if split_action[0] == "build" and split_action[1] == "road":
                return 10  # Road building reward
            elif split_action[0] == "build" and split_action[1] == "settlement":
                return 30  # Settlement building reward
            elif split_action[0] == "build" and split_action[1] == "city":
                return 35  # City building reward
            else:
                return 0  # Else, no reward

    def get_exploration_rate(self):
        return self.epsilon

    def set_exploration_rate(self, epsilon):
        pass
