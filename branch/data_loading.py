"""Load the competition's train/test CSVs."""
import pandas as pd


def load_data(data_dir="."):
    """Load train.csv and test.csv from data_dir.

    Returns
    -------
    train : pd.DataFrame, shape (80000, 18) -- includes 'target'
    test  : pd.DataFrame, shape (20000, 17) -- no 'target' column
    """
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    return train, test


if __name__ == "__main__":
    train, test = load_data()
    print(f"train: {train.shape}, test: {test.shape}")
    print(train.head())
