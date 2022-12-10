import pandas as pd

from helpers.profiling import timing


@timing
def read_csv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


@timing
def read_feather(path: str) -> pd.DataFrame:
    return pd.read_feather(path)
