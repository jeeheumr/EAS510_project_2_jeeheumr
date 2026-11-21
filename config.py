from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "vehicles.csv"
RESULTS = ROOT / "results" / "full"
SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5
FEATURES_A = ["displ", "cylinders"]
TARGET = "comb08"
