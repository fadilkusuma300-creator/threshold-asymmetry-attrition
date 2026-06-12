"""
Data loading module: load datasets from local CSV files.

Supported datasets:
  - IBM HR Attrition (primary experiment)
  - Indian HR Attrition (generalization validation)
  - Ziya07 Attrition (tertiary dataset)
"""
import os
import pandas as pd
from config import IBM_HR_FILE, INDIAN_HR_FILE, ZIYA07_FILE


def load_ibm_hr() -> pd.DataFrame:
    """Load the IBM HR Attrition dataset (1470 samples, 35 attributes)."""
    if not os.path.exists(IBM_HR_FILE):
        raise FileNotFoundError(f"Data file not found: {IBM_HR_FILE}")
    df = pd.read_csv(IBM_HR_FILE)
    print(f"[INFO] IBM HR dataset loaded: {df.shape}")
    return df


def load_indian_hr() -> pd.DataFrame:
    """Load the Indian HR Attrition dataset (5000 samples, 22 attributes)."""
    if not os.path.exists(INDIAN_HR_FILE):
        raise FileNotFoundError(f"Data file not found: {INDIAN_HR_FILE}")
    df = pd.read_csv(INDIAN_HR_FILE)
    print(f"[INFO] Indian HR dataset loaded: {df.shape}")
    return df


def load_ziya07() -> pd.DataFrame:
    """Load the Ziya07 Attrition dataset (10000 samples, 26 attributes)."""
    if not os.path.exists(ZIYA07_FILE):
        raise FileNotFoundError(f"Data file not found: {ZIYA07_FILE}")
    df = pd.read_csv(ZIYA07_FILE)
    print(f"[INFO] Ziya07 dataset loaded: {df.shape}")
    return df


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        Dataset name. Options: 'ibm', 'indian', 'ziya07'
    """
    loaders = {
        "ibm": load_ibm_hr,
        "indian": load_indian_hr,
        "ziya07": load_ziya07,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(loaders.keys())}")
    return loaders[name]()


if __name__ == "__main__":
    for name in ["ibm", "indian", "ziya07"]:
        df = load_dataset(name)
        print(f"  Columns: {list(df.columns)[:10]}...")
        print()
