import os

# Get project root no matter where the script runs from
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)  # go one level up from src/

# Data folders
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_INTERIM = os.path.join(PROJECT_ROOT, "data", "interim")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

# Models folder
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Create folders if missing
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_INTERIM, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
