import datetime as dt
from typing import Iterable, List, Union

import numpy as np
import pandas as pd


def filter_levels(series: pd.Series, min_threshold: float = 0.9) -> pd.Index:
    """
    Keep all the levels of the series such that
    the proportion of the kept levels exceeds
    the min_threshold.

    Args:
        series: categorical pd.series with levels
        threshold:
    Returns:
        pd.Index with the levels to keep
    """
    normalized_cumulative_counts = series.value_counts(normalize=True).cumsum()

    mask = normalized_cumulative_counts <= min_threshold
    return normalized_cumulative_counts[mask].index


def substitute_levels(
    series: pd.Series,
    levels_to_keep: Iterable,
    substitute_value: Union[str, int] = "other",
) -> pd.Series:
    """
    Args:
        series: pd.Series which represents a categorical variable
        levels_to_keep: list-like object with levels to keep
    Returns:
        series with levels_to_keep and/or substitute value
    """

    return series.where(cond=series.isin(levels_to_keep), other=substitute_value)


def cut_levels(
    series: pd.Series, min_threshold: float = 0.9, substitute_value="other"
) -> pd.Series:

    levels_to_keep = filter_levels(series, min_threshold)
    series_substituted = substitute_levels(
        series, levels_to_keep, substitute_value=substitute_value
    )

    return series_substituted


def extract_hour_str(time: str, time_format: str = "%H:%M:%S") -> int:
    return dt.datetime.strptime(time, time_format).hour


def extract_weekday_timestamp(
    date: pd.Timestamp,
) -> int:
    return date.dayofweek


def extract_month_datetime_timestamp(
    date: pd.Timestamp,
) -> int:
    return date.month


def extract_year_datetime_timestamp(
    date: pd.Timestamp,
) -> int:
    return date.year


def is_weekend(weekday: int, weekend_days: List[int] = [5, 6]) -> int:
    return int(weekday in weekend_days)


def create_datetime(row):
    return pd.Timestamp(int(row[0]), int(row[1]), int(row[2]))


def remove_dollar_sign(x: str) -> float:
    return float(x[1:])


def create_sin(x: int, period: int) -> float:
    return np.sin(2.0 * np.pi * x / period)


def create_cos(x: int, period: int) -> float:
    return np.cos(2.0 * np.pi * x / period)
