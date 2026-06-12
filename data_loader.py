"""
数据加载模块：从本地 CSV 文件加载数据集。

支持三个数据集：
  - IBM HR Attrition（主实验）
  - Indian HR Attrition（泛化验证）
  - Ziya07 Attrition（第三数据集）
"""
import os
import pandas as pd
from config import IBM_HR_FILE, INDIAN_HR_FILE, ZIYA07_FILE


def load_ibm_hr() -> pd.DataFrame:
    """加载 IBM HR Attrition 数据集（1470 样本，35 属性）。"""
    if not os.path.exists(IBM_HR_FILE):
        raise FileNotFoundError(f"数据文件不存在: {IBM_HR_FILE}")
    df = pd.read_csv(IBM_HR_FILE)
    print(f"[INFO] IBM HR 数据集已加载: {df.shape}")
    return df


def load_indian_hr() -> pd.DataFrame:
    """加载 Indian HR Attrition 数据集（5000 样本，22 属性）。"""
    if not os.path.exists(INDIAN_HR_FILE):
        raise FileNotFoundError(f"数据文件不存在: {INDIAN_HR_FILE}")
    df = pd.read_csv(INDIAN_HR_FILE)
    print(f"[INFO] Indian HR 数据集已加载: {df.shape}")
    return df


def load_ziya07() -> pd.DataFrame:
    """加载 Ziya07 Attrition 数据集（10000 样本，26 属性）。"""
    if not os.path.exists(ZIYA07_FILE):
        raise FileNotFoundError(f"数据文件不存在: {ZIYA07_FILE}")
    df = pd.read_csv(ZIYA07_FILE)
    print(f"[INFO] Ziya07 数据集已加载: {df.shape}")
    return df


def load_dataset(name: str) -> pd.DataFrame:
    """
    按名称加载数据集。

    Parameters
    ----------
    name : str
        数据集名称，可选: 'ibm', 'indian', 'ziya07'
    """
    loaders = {
        "ibm": load_ibm_hr,
        "indian": load_indian_hr,
        "ziya07": load_ziya07,
    }
    if name not in loaders:
        raise ValueError(f"未知数据集 '{name}'，可选: {list(loaders.keys())}")
    return loaders[name]()


if __name__ == "__main__":
    for name in ["ibm", "indian", "ziya07"]:
        df = load_dataset(name)
        print(f"  列名: {list(df.columns)[:10]}...")
        print()
