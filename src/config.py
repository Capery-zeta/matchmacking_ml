 
from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "matchmaking_dataset.csv"

# Target
TARGET_COLUMN = "relationship_longevity_months"

# Classification bins
CLASS_BINS = [0, 12, 36, 60, 120, float("inf")]
CLASS_LABELS = ["<1y", "1-3y", "3-5y", "5-10y", "10y+"]

# Reproducibility
RANDOM_STATE = 42