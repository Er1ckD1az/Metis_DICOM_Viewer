import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories (relative to project root)
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_DIR = DATA_DIR / "MICCAI_BraTS2020_TrainingData"
VAL_DATA_DIR = DATA_DIR / "MICCAI_BraTS2020_ValidationData"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def print_config():
    print("CONFIGURATION")
    print(f"Project Root:      {PROJECT_ROOT}")
    print(f"Training Data:     {TRAIN_DATA_DIR}")
    print(f"Validation Data:   {VAL_DATA_DIR}")
    print(f"Checkpoints:       {CHECKPOINT_DIR}")
    print(f"Results:           {RESULTS_DIR}")