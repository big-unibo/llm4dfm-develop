from pathlib import Path
from utils import load_yaml, load_ground_truth_exercise

input_config = load_yaml(f'{Path().absolute()}/pipeline/second-step-config.yml')



ground_truth = load_ground_truth_exercise()
