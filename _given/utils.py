
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def plot_value_function(value_function, maze, ax=None, title=None):
    """
    Plots a heatmap of the value function on the given Axes.
    """
    mask = (maze == 1) | (maze == 2) | (maze == 3)
    trap_positions = np.transpose(np.where(maze == 2))
    goal_positions = np.transpose(np.where(maze == 3))
    obs_positions = np.transpose(np.where(maze == 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    cmap = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256)
    sns.heatmap(
        value_function,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        cbar=False,
        linewidths=1,
        linecolor='black',
        ax=ax
    )
    for y, x in goal_positions:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, edgecolor='black', facecolor='darkgreen'))
    for y, x in trap_positions:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, edgecolor='black', facecolor='darkred'))
    for y, x in obs_positions:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, edgecolor='black', facecolor='gray'))
    if title:
        ax.set_title(title)


def plot_policy(value_function, maze, ax=None, title=None):
    """
    Plots policy arrows derived from the value function on the given Axes.
    """
    policy_arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    mask = (maze == 1) | (maze == 2) | (maze == 3)

    policy_grid = np.full(maze.shape, '', dtype='<U1')
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if mask[i, j]:
                continue
            best_val = -np.inf
            best_act = None
            for name, move in actions.items():
                ni, nj = i + move[0], j + move[1]
                if 0 <= ni < maze.shape[0] and 0 <= nj < maze.shape[1]:
                    val = value_function[ni, nj]
                    if val > best_val:
                        best_val, best_act = val, name
            if best_act:
                policy_grid[i, j] = policy_arrows[best_act]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    cmap = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256)
    sns.heatmap(
        value_function,
        mask=mask,
        annot=policy_grid,
        fmt="",
        cmap=cmap,
        cbar=False,
        linewidths=1,
        linecolor='black',
        ax=ax
    )
    trap_positions = np.transpose(np.where(maze == 2))
    goal_positions = np.transpose(np.where(maze == 3))
    obs_positions = np.transpose(np.where(maze == 1))
    for y, x in goal_positions:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, edgecolor='black', facecolor='darkgreen'))
    for y, x in trap_positions:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, edgecolor='black', facecolor='darkred'))
    for y, x in obs_positions:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, edgecolor='black', facecolor='gray'))
    if title:
        ax.set_title(title)


def plot_convergence(convergence, ax=None, title=None):
    """
    Plots the convergence curve (sum of absolute utility changes per episode).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(convergence) + 1), convergence)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Sum of |ΔU|')
    ax.grid(True)
    if title:
        ax.set_title(title)
    # No plt.show(), figure saving handled externally



import json
import matplotlib.pyplot as plt

def plot_learning_curves(json_paths, labels=None,  output_file="learning_curves.png"):
    """
    Plots both episode_rewards and average_scores from multiple experiment JSON files as subplots.
    
    Parameters:
        json_paths (list of str): List of file paths to JSON result files.
        labels (list of str, optional): Labels for each experiment (for legend).
                                        If not provided, file names (without .json) are used.
        output_file (str): Filename for saving the output image (e.g., "learning_curves.png").
        
    The function creates two subplots:
      - The first subplot shows raw episode_rewards vs. Episode number.
      - The second subplot shows average_scores (e.g., a 100-episode moving average) vs. Episode number.
    The figure is saved to the specified output_file and also displayed.
    """
    # Load data from JSON files
    data_list = []
    for path in json_paths:
        with open(path, 'r') as f:
            data_list.append(json.load(f))
            
    # Generate default labels from file names if none are provided
    if labels is None:
        labels = [path.split('/')[-1].replace('.json', '') for path in json_paths]
    
    # Create figure with two subplots (vertical layout)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Subplot 1: Raw episode rewards
    axs[0].set_title("Episode Rewards")
    for idx, data in enumerate(data_list):
        if "episode_rewards" not in data:
            raise KeyError(f"Key 'episode_rewards' not found in {json_paths[idx]}.")
        rewards = data["episode_rewards"]
        episodes = range(1, len(rewards) + 1)
        axs[0].plot(episodes, rewards, label=labels[idx])
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    axs[0].grid(True)
    
    # Subplot 2: Moving average scores
    axs[1].set_title("Average Scores (Moving Average)")
    for idx, data in enumerate(data_list):
        if "average_scores" not in data:
            raise KeyError(f"Key 'average_scores' not found in {json_paths[idx]}.")
        avg_scores = data["average_scores"]
        episodes = range(1, len(avg_scores) + 1)
        axs[1].plot(episodes, avg_scores, label=labels[idx])
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Average Score")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)  # Save the figure to an image file
    plt.show()

def plot_solved_episodes(json_paths, labels=None, output_file="solved_episodes.png"):
    """
    Plot a bar chart showing the episode number at which each experiment solved the task.
    
    Parameters:
        json_paths (list of str): List of JSON result file paths (each corresponding to an experiment).
        labels (list of str, optional): Names for each experiment (for x-axis labels). If not provided, file names will be used.
        output_file (str): Filename for saving the output image (e.g., "solved_episodes.png").
    
    Each JSON file is expected to have a 'solved_episode' entry (the episode index when the solve criterion was first met).
    Experiments that did not solve within the training duration are omitted from the chart.
    """
    solved_eps = []
    exp_names = []
    for i, path in enumerate(json_paths):
        with open(path, 'r') as f:
            data = json.load(f)
        episode = data.get("solved_episode", None)
        if episode is not None and episode is not False:
            solved_eps.append(episode)
            # Determine label for this bar
            if labels and i < len(labels):
                exp_names.append(labels[i])
            else:
                exp_names.append(path.split('/')[-1].replace('.json',''))
        # If not solved (episode is None or False/ -1), skip this experiment for the bar chart
    if not solved_eps:
        print("No experiments solved the environment to plot.")
        return
    plt.figure(figsize=(8,5))
    colors = plt.get_cmap('tab20').colors  # get a list of colors
    bars = plt.bar(exp_names, solved_eps, color=[colors[i % len(colors)] for i in range(len(solved_eps))])
    plt.ylabel('Episode Solved')
    plt.title('Solved Episode by Experiment')
    plt.xticks(rotation=45, ha='right')
    # Annotate each bar with the episode number
    for rect, ep in zip(bars, solved_eps):
        plt.text(rect.get_x() + rect.get_width()/2, ep + 2, str(ep), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(output_file)  # Save the figure to an image file
    plt.show()