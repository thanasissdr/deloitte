import numpy as np
import pandas as pd


def describe(series: pd.Series, fmt=".3f") -> pd.Series:
    if np.issubdtype(series.dtype, (np.float_)):
        return series.describe().map(f"{{:{fmt}}}".format)
    else:
        raise ValueError(f"{series.dtype} is not supported")
