import gymnasium as gym
import numpy as np
import torch
from collections import deque
import random
import json

from models import QNetwork_1, QNetwork_2, QNetwork_3, QNetwork_4, QNetwork_5

#  Define a replay memory to store experiences:
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

#  Define the DQN Agent encapsulating the networks, memory, and training procedure:
class DQNAgent:
    def __init__(self, 
                 state_dim, 
                 action_dim,
                 Net_No=3,
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
        self.alpha = alpha
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = ReplayMemory(memory_size)

        match Net_No:
            case 1:
                self.policy_net = QNetwork_1(state_dim, action_dim)
                self.target_net = QNetwork_1(state_dim, action_dim)
            case 2:
                self.policy_net = QNetwork_2(state_dim, action_dim)
                self.target_net = QNetwork_2(state_dim, action_dim)
            case 3:
                self.policy_net = QNetwork_3(state_dim, action_dim)
                self.target_net = QNetwork_3(state_dim, action_dim)
            case 4:
                self.policy_net = QNetwork_4(state_dim, action_dim)
                self.target_net = QNetwork_4(state_dim, action_dim)
            case 5:
                self.policy_net = QNetwork_5(state_dim, action_dim)
                self.target_net = QNetwork_5(state_dim, action_dim)
            case _:
                print("Error on Net No: missing")

        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.target_update_freq = target_update_freq
        self.solved_score = 200.0       #target average reward for ’solved’
        self.solved_window = 100        #number of episodes to average for solved check

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
        curr_Q = self.policy_net(states).gather(1, actions)
        next_Q = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        
        # Compute target Q
        target_Q = rewards + (1 - dones) * (self.gamma * next_Q)
        #target_Q = curr_Q + self.alpha * (rewards + self.gamma * next_Q - curr_Q)

        # Compute loss
        loss = torch.nn.functional.mse_loss(curr_Q, target_Q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        # Update target network to match policy network.
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        # Decay exploration rate after each episode.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def run_experiment(config, output_path):
    # Main training loop for a single experiment
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    #print(f"lr={config['lr']} | g={config['gamma']} | ed={config['epsilon_decay']} | f={config['target_update_freq']} | NN={config['net_no']}")
    
    agent = DQNAgent(state_dim, 
                     action_dim, 
                     Net_No=config['net_no'],
                     gamma=config['gamma'],
                     epsilon_decay=config['epsilon_decay'],
                     target_update_freq=config['target_update_freq'],
                    )
    num_episodes = config['num_episodes']

    rewards_history = []
    avg_scores = []
    solved_episode = None
    scores_window = deque(maxlen=agent.solved_window)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        for t in range(1000): # limit max steps per episode to avoid long runs
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
            if done:
                break
            
        # Episode end
        rewards_history.append(total_reward)
        scores_window.append(total_reward)
        # Compute moving average
        avg = np.mean(scores_window)
        avg_scores.append(avg)
        # Check solved
        if solved_episode is None and len(scores_window) == agent.solved_window and avg >= agent.solved_score:
            solved_episode = episode
        # Epsilon decay
        agent.decay_epsilon()
        # Update target network
        if episode % agent.target_update_freq == 0:
            agent.update_target()
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Avg: {avg:.2f} | Epsilon: {agent.epsilon:.3f}")
    
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
    base_config = {
        'lr': 1e-3,
        'gamma': 0.99,
        'epsilon_decay': 0.995,
        'target_update_freq': 10,
        'net_no': 3,
        'num_episodes': 100
    }

    param_grid = {
        'lr': [1e-4, 1e-3, 5e-3],
        'gamma': [0.98, 0.99, 0.999],
        'epsilon_decay': [0.98, 0.99, 0.995],
        'target_update_freq': [1, 10, 50],
        'net_no': [1, 2, 3, 4, 5]
    }

    for param, values in param_grid.items():
        for val in values:
            config = base_config.copy()
            config[param] = val
            save_path = f'part2/results/{param}_{val}.json'
            print(f"Running experiment with {param}={val}")
            run_experiment(config, save_path)
