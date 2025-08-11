import torch
import torch.nn as nn # Added missing import
import torch.optim as optim
import numpy as np
import random
from collections import deque
from .dqn_model import DQN

class DQNAgent:
    """
    Deep Q-Network (DQN) Agent.

    This class implements the core logic for a DQN agent, including
    experience replay, Q-value estimation, and target network updates.
    """
    def __init__(self, input_shape, num_actions, learning_rate, gamma, memory_size, epsilon_start, epsilon_end, epsilon_decay):
        """
        Initializes the DQNAgent.

        Args:
            input_shape (tuple): Shape of the input state (e.g., (4, 84, 84) for stacked frames).
            num_actions (int): Number of possible actions the agent can take.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            memory_size (int): Maximum size of the replay memory.
            epsilon_start (float): Initial value of epsilon for epsilon-greedy policy.
            epsilon_end (float): Minimum value of epsilon.
            epsilon_decay (float): Decay rate for epsilon.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}") # Translated message

        self.policy_net = DQN(input_shape, num_actions).to(self.device)
        self.target_net = DQN(input_shape, num_actions).to(self.device)
        self.update_target_network()
        self.target_net.eval() # Set target network to evaluation mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size) # Experience replay memory
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.num_actions = num_actions
        self.gamma = gamma

    def update_target_network(self):
        """
        Updates the target network's weights with the policy network's weights.
        This is done periodically to stabilize training.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple (state, action, reward, next_state, done) in the replay memory.

        Args:
            state (numpy.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (numpy.ndarray): The next state.
            done (bool): True if the episode ended, False otherwise.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (numpy.ndarray): The current state.

        Returns:
            int: The chosen action.
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.num_actions)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_values = self.policy_net(state_tensor)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        """
        Performs a Q-learning update using a batch of experiences from the replay memory.

        Args:
            batch_size (int): The number of experiences to sample from memory.

        Returns:
            float or None: The loss value if a replay is performed, otherwise None.
        """
        if len(self.memory) < batch_size:
            return None

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Compute Q-values for current states
        current_q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q-values for next states
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return loss.item()