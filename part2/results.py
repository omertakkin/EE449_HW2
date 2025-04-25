import sys
import os
sys.path.append(os.path.abspath('_given'))
from utils import plot_learning_curves, plot_solved_episodes # type: ignore


def main():
    # Directory containing experiment JSON results
    base_dir = os.path.join(os.path.dirname(__file__), 'results')

    # Hyperparameter experiments and their values (must match JSON filenames)
    experiments = {
        'lr': [0.0001, 0.001, 0.005],
        'gamma': [0.98, 0.99, 0.999],
        'epsilon_decay': [0.98, 0.99, 0.995],
        'target_update_freq': [1, 10, 50],
        'net_no': [1, 2, 3, 4, 5]
    }

    # Plot learning curves for each hyperparameter
    for param, values in experiments.items():
        # Construct list of JSON paths for this parameter
        json_paths = []
        labels = []
        for val in values:
            # Format value to match filename
            val_str = str(val)
            json_file = f"{param}_{val_str}.json"
            path = os.path.join(base_dir, json_file)
            if os.path.isfile(path):
                json_paths.append(path)
                labels.append(str(val))
            else:
                print(f"Warning: missing result file {path}")

        if not json_paths:
            print(f"No data for parameter '{param}', skipping.")
            continue

        # Output image filename
        out_png = f"part2/results_fig/learning_curves_{param}.png"
        # Generate and save figure
        plot_learning_curves(
            json_paths=json_paths,
            labels=labels,
            output_file=out_png
        )
        print(f"Saved learning curves for '{param}' at: {out_png}")

    # Plot bar chart of solved episodes for all experiments
    # Gather all JSON result files
    all_json = []
    for fname in os.listdir(base_dir):
        if fname.endswith('.json'):
            all_json.append(os.path.join(base_dir, fname))

    if not all_json:
        print("No JSON result files found, skipping solved-episode bar chart.")
        return
    
    all_labels = [
        os.path.splitext(os.path.basename(path))[0]
        for path in all_json
    ]

    # Output bar chart filename
    out_bar = 'part2/results_fig/solved_episodes.png'

    # Generate and save bar chart
    plot_solved_episodes(
        json_paths=all_json,
        labels = all_labels,
        output_file=out_bar
    )
    print(f"Saved solved episodes bar chart at: {out_bar}")


if __name__ == '__main__':
    main()
