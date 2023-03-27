from functools import wraps
from typing import Any

import numpy as np
import pandas as pd


def apply_function_to_column_elementwise(f):
    @wraps(f)
    def inner(data: Any, **kwargs):
        if isinstance(data, pd.DataFrame):
            val = data.applymap(f, **kwargs).values
        elif isinstance(data, pd.Series):
            val = data.apply(f, **kwargs).values.reshape(-1, 1)
        elif isinstance(data, np.ndarray):
            vectorized_f  = np.vectorize(lambda x: f(x, **kwargs))
            val = vectorized_f(data)
        else:
            raise ValueError(f"{type(data)} is not supported.")
        return val

    return inner


def apply_function_to_row(f):
    @wraps(f)
    def inner(data: Any, **kwargs):
        if isinstance(data, pd.DataFrame):
            val = data.apply(f, axis=1, **kwargs).values.reshape(-1, 1)
        elif isinstance(data, np.ndarray):
            val = np.array(list(map(lambda x: f(x, **kwargs), data))).reshape(-1, 1)
        else:
            raise ValueError(f"{type(data)} is not supported.")
        
        return val

    return inner
