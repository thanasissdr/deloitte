import numpy as np
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp

from pipelines import preprocessing_base
from pipelines.decorators import (
    apply_function_to_column_elementwise,
    apply_function_to_row,
)


def frequency_encoding(x: pd.Series, normalize: bool = False) -> np.array:
    tmp = x.value_counts(normalize=normalize)
    return x.map(tmp).values.reshape(-1, 1)


@apply_function_to_column_elementwise
def extract_hour_str(time: str, **kwargs) -> int:
    return preprocessing_base.extract_hour_str(time, **kwargs)


@apply_function_to_column_elementwise
def extract_weekday_timestamp(timestamp: Timestamp) -> int:
    return preprocessing_base.extract_weekday_timestamp(timestamp)


@apply_function_to_column_elementwise
def is_weekend(weekday: int, **kwargs):
    return preprocessing_base.is_weekend(weekday, **kwargs)


@apply_function_to_row
def create_datetime(row, **kwargs):
    return preprocessing_base.create_datetime(row, **kwargs)


@apply_function_to_column_elementwise
def create_sin(x: int, **kwargs):
    return preprocessing_base.create_sin(x, **kwargs)


@apply_function_to_column_elementwise
def create_cos(x: int, **kwargs):
    return preprocessing_base.create_cos(x, **kwargs)


@apply_function_to_column_elementwise
def remove_dollar_sign(x):
    return preprocessing_base.remove_dollar_sign(x)


@apply_function_to_row
def create_datetime(row):
    return preprocessing_base.create_datetime(row[0], row[1], row[2])


@apply_function_to_column_elementwise
def to_pd_datetime(x: np.datetime64):
    return pd.Timestamp(x)
