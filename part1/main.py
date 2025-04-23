import sys
import os
sys.path.append(os.path.abspath('_given'))

import numpy as np
import matplotlib.pyplot as plt
from utils import plot_value_function, plot_policy, plot_convergence  # type: ignore

import gc

# Ensure results directory exists
os.makedirs('part1/results', exist_ok=True)

class MazeEnvironment:
    def __init__(self):
        # Define the maze layout, rewards, and action space
        self.start_pos = (0, 0)
        self.current_pos = self.start_pos
        self.state_penalty = -1
        self.trap_penalty = -100
        self.goal_reward = 100
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        # Transition probabilities
        self.prob_intended = 0.75
        self.prob_opposite = 0.05
        self.prob_perpendicular = 0.10

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action):
        r = np.random.rand()
        if r < self.prob_intended:
            chosen = action
        elif r < self.prob_intended + self.prob_opposite:
            chosen = {0:1, 1:0, 2:3, 3:2}[action]
        elif r < self.prob_intended + self.prob_opposite + self.prob_perpendicular:
            chosen = {0:2, 1:3, 2:1, 3:0}[action]
        else:
            chosen = {0:3, 1:2, 2:0, 3:1}[action]

        move = self.actions[chosen]
        next_state = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])
        if (0 <= next_state[0] < self.maze.shape[0] and 0 <= next_state[1] < self.maze.shape[1]
                and self.maze[next_state] != 1):
            self.current_pos = next_state

        cell_value = self.maze[self.current_pos]
        if cell_value == 2:
            return self.current_pos, self.trap_penalty, True
        elif cell_value == 3:
            return self.current_pos, self.goal_reward, True
        else:
            return self.current_pos, self.state_penalty, False

class MazeTD0(MazeEnvironment):
    def __init__(self, maze, alpha=0.001, gamma=0.95, epsilon=0.2, episodes=10000):
        super().__init__()
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        # Initialize utilities
        self.utility = np.random.rand(*self.maze.shape) * 0.1
        self.utility[maze == 1] = np.nan

    def choose_action(self, state):
        valid = []
        for a, move in self.actions.items():
            ns = (state[0] + move[0], state[1] + move[1])
            if 0 <= ns[0] < self.maze.shape[0] and 0 <= ns[1] < self.maze.shape[1] and self.maze[ns] != 1:
                valid.append(a)
        if not valid:
            return None
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid)
        best, best_val = None, -np.inf
        for a in valid:
            move = self.actions[a]
            ns = (state[0] + move[0], state[1] + move[1])
            val = self.utility[ns]
            if val > best_val:
                best, best_val = a, val
        return best

    def update_utility_value(self, s, r, ns):
        curr = self.utility[s]
        new = self.utility[ns]
        self.utility[s] = curr + self.alpha * (r + self.gamma * new - curr)

    def run_episodes(self, snapshot_episodes=None, max_steps=5000000):
        if snapshot_episodes is None:
            snapshot_episodes = []
        snapshots = {}
        convergence = np.zeros(self.episodes)
        
        reached_step = 0 # checking if we hit max_steps

        for ep in range(1, self.episodes + 1):
            state = self.reset()
            diff = 0
            steps = 0
            done = False

            while not done and steps < max_steps:
                act = self.choose_action(state)
                if act is None:
                    break
                old = self.utility[state]
                next_state, reward, done = self.step(act)
                self.update_utility_value(state, reward, next_state)
                diff += abs(self.utility[state] - old)
                state = next_state
                steps += 1
            
            reached_step = max(reached_step , steps)
            convergence[ep - 1] = diff
            if ep in snapshot_episodes:
                snapshots[ep] = self.utility.copy()
        print(f"Reached maximum step is: {reached_step}")
        return snapshots, convergence
    


def save_results(param, val, snapshots, convergence):
    base = f"part1/results/{param}_{val}"
    os.makedirs(base, exist_ok=True)

    for ep, u in snapshots.items():
        np.save(f"{base}/utility_ep{ep}.npy", u)

    np.save(f"{base}/convergence.npy", np.array(convergence))



if __name__ == '__main__':
    # Define maze
    maze = np.array([
        [0,0,0,0,0,0,0,1,1,1,1],
        [1,0,1,1,0,1,0,0,0,0,1],
        [1,0,1,1,1,1,0,1,1,0,1],
        [1,0,0,0,0,1,0,0,1,0,1],
        [1,1,0,1,0,1,0,1,1,0,0],
        [1,1,0,1,0,0,0,0,1,1,0],
        [1,0,0,1,1,0,1,0,0,0,0],
        [0,0,1,1,1,1,2,0,1,1,0],
        [0,0,0,0,0,0,0,1,0,0,0],
        [1,0,1,0,1,1,0,1,1,1,0],
        [1,1,1,0,2,0,1,0,0,3,0]
    ])

    # Hyperparameter experiments
    experiments = {
        'alpha': {'values': [0.001, 0.01, 0.1, 0.5, 1.0], 'default': 0.1},
        'gamma': {'values': [0.10, 0.25, 0.50, 0.75, 0.95], 'default': 0.95},
        'epsilon': {'values': [0, 0.2, 0.5, 0.8, 1.0], 'default': 0.2}
    }
    episodes = 10000
    snapshots_to_keep = [1, 50, 100, 1000, 5000, 10000]

    for param, config in experiments.items():
        for val in config['values']:
            # Set parameters
            alpha = experiments['alpha']['default'] if param != 'alpha' else val
            gamma = experiments['gamma']['default'] if param != 'gamma' else val
            epsilon = experiments['epsilon']['default'] if param != 'epsilon' else val

            print(f"Selected values are: alpha = {alpha} | gamma = {gamma} | epsilon = {epsilon} ")

            # Run TD(0)
            agent = MazeTD0(maze, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes)
            snapshots, convergence = agent.run_episodes(snapshots_to_keep)

                        # Plot and save value function snapshots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            for ax, ep in zip(axes, snapshots_to_keep):
                plot_value_function(snapshots[ep], maze, ax=ax, title=f"{param}={val}, Ep={ep}")
            fig.suptitle(f"Value Function Evolution for {param}={val}")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(f"part1/results/{param}_{val}_value_evolution.png")
            plt.close(fig)

            # Plot and save policy snapshots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            for ax, ep in zip(axes, snapshots_to_keep):
                plot_policy(snapshots[ep], maze, ax=ax, title=f"{param}={val}, Ep={ep}")
            fig.suptitle(f"Policy Evolution for {param}={val}")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(f"part1/results/{param}_{val}_policy_evolution.png")
            plt.close(fig)

            # Plot and save convergence
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_convergence(convergence, ax=ax, title=f"Convergence for {param}={val}")
            fig.savefig(f"part1/results/{param}_{val}_convergence.png")
            plt.close(fig)