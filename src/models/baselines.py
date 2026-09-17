import pandas as pd


def seasonal_naive_forecast(
    data: pd.DataFrame,
    lag_days: int = 7,
) -> pd.Series:
    """
    Forecast demand using sales from the same series
    `lag_days` earlier.

    A 7-day lag represents a weekly seasonal naive forecast.
    """
    lag_column = f"lag_{lag_days}"

    if lag_column not in data.columns:
        raise ValueError(
            f"Required feature '{lag_column}' is missing."
        )

    return data[lag_column].copy()


def rolling_mean_forecast(
    data: pd.DataFrame,
    window: int = 28,
) -> pd.Series:
    """
    Forecast demand using the historical rolling mean
    for each series.
    """
    rolling_column = f"rolling_mean_{window}"

    if rolling_column not in data.columns:
        raise ValueError(
            f"Required feature '{rolling_column}' is missing."
        )

    return data[rolling_column].copy()