import numpy as np
import pandas as pd


def get_value_counts(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame()

    for normalize, normalize_label in zip([False, True], ["absolute", "normalized"]):
        counts_series = series.value_counts(normalize=normalize)

        counts_series.name = f"{series.name}_{normalize_label}_count"
        counts_series_cumsum = counts_series.cumsum()
        counts_series_cumsum.name = f"{counts_series.name} _cumsum"
        df = pd.concat([df, counts_series, counts_series_cumsum], axis=1)

    return df


def describe(series: pd.Series) -> pd.Series:
    if series.dtype == "O" or np.issubdtype(series.dtype, (np.integer)):
        return series.astype(str).describe()
    else:
        raise ValueError(f"{series.dtype} is not supported")


def _get_contributions_of_features(
    feature_name: str, df: pd.DataFrame, errors_indicator_series: pd.Series
) -> pd.DataFrame:
    transactions_with_error = (
        df[feature_name][errors_indicator_series == 1]
        .value_counts(normalize=True)
        .reset_index(drop=False)
    )

    transactions_without_error = (
        df[feature_name][errors_indicator_series == 0]
        .value_counts(normalize=True)
        .reset_index(drop=False)
    )

    merged = transactions_with_error.merge(
        transactions_without_error, on="index", how="outer"
    )

    merged["diff"] = merged[f"{feature_name}_x"] - merged[f"{feature_name}_y"]

    merged = merged.sort_values("diff", ascending=False)
    return merged
