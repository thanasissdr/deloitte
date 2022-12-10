from functools import wraps

import pandas as pd


def apply_function_to_column_elementwise(f):
    @wraps(f)
    def inner(data: pd.Series | pd.DataFrame, **kwargs):
        if isinstance(data, pd.DataFrame):
            val = data.applymap(f, **kwargs).values
        elif isinstance(data, pd.Series):
            val = data.apply(f, **kwargs).values.reshape(-1, 1)
        else:
            raise ValueError(f"{type(data)} is not supported.")
        return val

    return inner


def apply_function_to_row(f):
    @wraps(f)
    def inner(df: pd.DataFrame, **kwargs):
        return df.apply(f, axis=1, **kwargs).values.reshape(-1, 1)

    return inner
