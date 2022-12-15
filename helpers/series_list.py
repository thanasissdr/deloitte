from typing import Iterable, List

import pandas as pd


def get_unique_values(series: pd.Series) -> set:
    """
    Args:
        series: pd.Series where each row is a string with comma separated values
    Returns:
        Set of unique values in all elements combined
    """
    unique_values = set()

    for i in series.values:
        if isinstance(i, (str,)):
            for j in i.split(","):
                if j not in unique_values and j.strip():
                    unique_values.add(j)
    return unique_values


def _set_bag_of_words_freq(iterable: Iterable, val: str) -> int:
    if iterable:
        return int(val in iterable)
    return 0


def bag_of_words_series(series: pd.Series) -> pd.DataFrame:
    """
    Args:
        series: pd.Series where each row is a comma separated values
    Returns:
        pd.DataFrame with extra columns which are the unique elements
        in the series and returns a dataframe which is similar
        to a bag of words approach.
    """

    unique_elements = get_unique_values(series)

    df = pd.DataFrame()

    for element in unique_elements:
        df[f"{series.name}_{element}"] = series.apply(
            _set_bag_of_words_freq, args=(element,)
        )

    return df


def _get_n_elements(x: str):
    if x is not None:
        eval_x: List = x.split(",")
        count = 0
        for i in eval_x:
            if i:
                count += 1

        return count

    return 0


def get_n_elements(series: pd.Series) -> pd.Series:
    """
    Args:
        series: pd.Series where each row is a string with comma separated values
    Returns:
        pd.Series with n_elements in each row
    """

    return series.apply(_get_n_elements)


