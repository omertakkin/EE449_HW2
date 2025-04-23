import sys
import os
sys.path.append(os.path.abspath('_given'))
from utils import plot_learning_curves, plot_solved_episodes # type: ignore # your utility plotting functions
import glob


# Define experiment groups and corresponding file patterns
experiment_groups = {
    'learning_rate': {
        'pattern': 'results_lr_*.json',
        'labels': ['lr=1e-4', 'lr=1e-3', 'lr=5e-3'],
        'output_curve': 'learning_curves_lr.png',
        'output_solved': 'solved_episodes_lr.png'
    },
    'discount_factor': {
        'pattern': 'results_gamma_*.json',
        'labels': ['gamma=0.98', 'gamma=0.99', 'gamma=0.999'],
        'output_curve': 'learning_curves_gamma.png',
        'output_solved': 'solved_episodes_gamma.png'
    },
    'epsilon_decay': {
        'pattern': 'results_decay_*.json',
        'labels': ['decay=0.98', 'decay=0.99', 'decay=0.995'],
        'output_curve': 'learning_curves_decay.png',
        'output_solved': 'solved_episodes_decay.png'
    },
    'target_update_freq': {
        'pattern': 'results_update_*.json',
        'labels': ['freq=1', 'freq=10', 'freq=50'],
        'output_curve': 'learning_curves_update.png',
        'output_solved': 'solved_episodes_update.png'
    },
    'architecture': {
        'pattern': 'results_net_*.json',
        'labels': ['128-64-64', '128-128', '256-256'],
        'output_curve': 'learning_curves_arch.png',
        'output_solved': 'solved_episodes_arch.png'
    }
}

if __name__ == '__main__':
    for name, cfg in experiment_groups.items():
        # Discover matching result files
        json_paths = sorted(glob.glob(cfg['pattern']))
        if not json_paths:
            print(f"No result files found for {name} (pattern {cfg['pattern']})")
            continue

        print(f"Plotting group: {name}")
        # Plot learning curves
        plot_learning_curves(
            json_paths,
            labels=cfg['labels'],
            output_file=cfg['output_curve']
        )
        print(f"  Saved learning curves to {cfg['output_curve']}")

        # Plot solved-episodes bar chart
        plot_solved_episodes(
            json_paths,
            labels=cfg['labels'],
            output_file=cfg['output_solved']
        )
        print(f"  Saved solved-episode chart to {cfg['output_solved']}\n")
