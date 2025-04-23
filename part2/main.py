import gymnasium as gym
import numpy as np
import torch
from collections import deque
import random
import json

# Q-Network definition
class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=128, hidden_dim2=128):
        super(QNetwork, self).__init__()
        # first hidden layer
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim1)
        # second hidden layer
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        # output layer to Q‑values
        self.fc3 = torch.nn.Linear(hidden_dim2, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay memory for experience replay
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# DQN Agent encapsulating training and action selection
class DQNAgent:
    def __init__(self, 
                 state_dim, 
                 action_dim,
                 memory_size=50000, 
                 batch_size=64,
                 gamma=0.99, 
                 alpha=1e-3,
                 epsilon_start=1.0, 
                 epsilon_min=0.01, 
                 epsilon_decay=0.995,
                 target_update_freq=10):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = ReplayMemory(memory_size)
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.target_update_freq = target_update_freq
        self.solved_score = 200.0
        self.solved_window = 100

    def get_action(self, state):
        # Select an action for the given state using epsilon-greedy policy.
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_v)
            return int(q_values.argmax(dim=1)[0])

    def train_step(self):
        #Perform one training step.
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to train yet.
        batch = self.memory.sample(self.batch_size)
        states = torch.tensor(np.array([exp[0] for exp in batch]), dtype=torch.float32)
        actions = torch.tensor(np.array([exp[1] for exp in batch]), dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(np.array([exp[2] for exp in batch]), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array([exp[3] for exp in batch]), dtype=torch.float32)
        dones = torch.tensor(np.array([exp[4] for exp in batch]), dtype=torch.float32).unsqueeze(1)

        # Current Q values
        curr_q = self.policy_net(states).gather(1, actions)
        # Next Q values from target network
        next_q = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        # Compute target Q
        target_q = rewards + (1 - dones) * (self.gamma * next_q)

        # Compute loss
        loss = torch.nn.functional.mse_loss(curr_q, target_q)
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        # Sync target network parameters
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        # Decay epsilon after each episode
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Main training loop for a single experiment

def run_experiment(config, output_path):
    # config: dict with hyperparameters
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    num_episodes = 10000

    rewards_history = []
    avg_scores = []
    solved_episode = None
    scores_window = deque(maxlen=agent.solved_window)

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
        # Episode end
        rewards_history.append(total_reward)
        scores_window.append(total_reward)
        # Compute moving average
        avg = np.mean(scores_window)
        avg_scores.append(avg)
        # Check solved
        if solved_episode is None and len(scores_window) == agent.solved_window and avg >= agent.solved_score:
            solved_episode = ep
        # Epsilon decay
        agent.decay_epsilon()
        # Update target network
        if ep % agent.target_update_freq == 0:
            agent.update_target()
        print(f"Episode {ep} | Reward: {total_reward:.2f} | Avg: {avg:.2f} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    # Save results
    results = {
        'episode_rewards': rewards_history,
        'average_scores': avg_scores,
        'hyperparameters': config,
        'solved_episode': solved_episode
    }
    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Define a sample configuration
    config = {
        'lr': 1e-3,
        'gamma': 0.99,
        'epsilon_decay': 0.995,
        'target_update_freq': 10,
        'num_episodes': 5000,
        'batch_size': 64,
        'memory_size': 50000
    }
    run_experiment(config, 'results_default.json')
