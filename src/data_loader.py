import pandas as pd
import yaml
import os

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_raw_data():
    """Loads the raw dataset from the data/raw directory."""
    raw_path = config["data"]["raw"]

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset not found at {raw_path}")
    
    df = pd.read_csv(raw_path)
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
    print(f"Dataset shape: Total rows in dataset are: {df.shape[0]} and total columns in dataset are: {df.shape[1]}")