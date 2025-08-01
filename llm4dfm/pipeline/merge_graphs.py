import os
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt

def collect_csvs(root_dir, include_dir=None, exclude_dir=None):
    csv_files = []

    # Walk through all directories and subdirectories
    for dirpath, dir_name, filenames in os.walk(root_dir):
        if (not include_dir or include_dir(dirpath)) and (not exclude_dir or not exclude_dir(dirpath)):
            for file in filenames:
                if file.endswith(".csv"):
                    csv_path = os.path.join(dirpath, file)
                    csv_files.append(csv_path)
        else:
            print(f'Skipping {dirpath}')

    if not csv_files:
        print("No CSV files found.")
        return

    dataframes = []
    for file in csv_files:
        try:
            dataframes.append(pd.read_csv(file))
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    return pd.concat(dataframes, ignore_index=True)



def plot_f1_vs_time_per_exercise(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Group by exercise
    for ex_name, ex_data in data.groupby("ex_name"):
        # Compute averages per config_label
        grouped = ex_data.groupby("config_label").agg({
            "nodes_f1": "mean",
            "edges_f1": "mean",
            "time": "mean"
        }).reset_index()

        plt.figure(figsize=(10, 6))

        # Scatter plot
        plt.scatter(grouped["time"], grouped["nodes_f1"], color='black', label="Nodes F1", alpha=0.7)
        plt.scatter(grouped["time"], grouped["edges_f1"], color='red', label="Edges F1", alpha=0.7)

        # Annotate each point with config label
        for _, row in grouped.iterrows():
            plt.text(row["time"], row["nodes_f1"], row["config_label"],
                     fontsize=6, ha="left", va="bottom", color='black', rotation=30)
            plt.text(row["time"], row["edges_f1"], row["config_label"],
                     fontsize=6, ha="left", va="bottom", color='red', rotation=30)

        plt.title(f"{ex_name} – F1 vs. Time per Config")
        plt.xlabel("Average Elapsed Time (s)")
        plt.ylabel("Average F1-score")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()

        # Save plot
        safe_ex_name = ex_name.replace(" ", "_").replace("/", "_")
        filename = os.path.join(output_dir, f"{safe_ex_name}_f1_vs_time.pdf")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        print(f"Saved plot for '{ex_name}' to {filename}")

load_dotenv()

root_directory = os.getenv('RESULTS')

def exclude(dir_name):
    return dir_name.endswith('paper')

merged_df = collect_csvs(root_directory, exclude_dir=exclude)

plot_f1_vs_time_per_exercise(merged_df, root_directory)