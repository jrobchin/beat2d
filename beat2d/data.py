import pandas as pd

from beat2d import settings


def get_dataset(path: str = settings.DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
