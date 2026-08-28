import numpy as np
import pandas as pd


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features for demand forecasting.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed M5 data containing a date column.

    Returns
    -------
    pd.DataFrame
        Data with additional calendar features.
    """
    data = data.copy()

    data["date"] = pd.to_datetime(data["date"])

    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_month"] = data["date"].dt.day
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["quarter"] = data["date"].dt.quarter

    data["is_weekend"] = (
        data["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    return data
def add_cyclical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical encodings for periodic calendar variables.
    """
    data = data.copy()

    data["dow_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["dow_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    return data
def add_event_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary event indicators from the M5 calendar fields.
    """
    data = data.copy()

    data["has_event_1"] = data["event_name_1"].notna().astype(int)
    data["has_event_2"] = data["event_name_2"].notna().astype(int)

    data["has_any_event"] = (
        (data["has_event_1"] == 1)
        | (data["has_event_2"] == 1)
    ).astype(int)

    return data

    return data
def add_snap_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add the applicable SNAP indicator for each store's state.
    """
    data = data.copy()

    data["snap"] = 0

    data.loc[
        data["state_id"] == "CA",
        "snap",
    ] = data.loc[
        data["state_id"] == "CA",
        "snap_CA",
    ]

    data.loc[
        data["state_id"] == "TX",
        "snap",
    ] = data.loc[
        data["state_id"] == "TX",
        "snap_TX",
    ]

    data.loc[
        data["state_id"] == "WI",
        "snap",
    ] = data.loc[
        data["state_id"] == "WI",
        "snap_WI",
    ]

    data["snap"] = data["snap"].astype(int)

    return data
def add_price_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-based features using recent price history.
    """
    data = data.copy()

    group_columns = [
        "store_id",
        "cat_id",
    ]

    data = data.sort_values(
        group_columns + ["date"]
    )

    grouped_price = data.groupby(
        group_columns,
        observed=True,
    )["avg_sell_price"]

    data["price_change_1"] = (
        grouped_price
        .diff(1)
        .fillna(0)
        .astype("float32")
    )

    data["price_change_7"] = (
        grouped_price
        .diff(7)
        .fillna(0)
        .astype("float32")
    )

    data["price_mean_28"] = (
        grouped_price
        .transform(
            lambda s: s.rolling(
                28,
                min_periods=1,
            ).mean()
        )
        .astype("float32")
    )

    data["price_vs_mean_28"] = (
        data["avg_sell_price"]
        / data["price_mean_28"].replace(0, np.nan)
    )

    data["price_vs_mean_28"] = (
        data["price_vs_mean_28"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .astype("float32")
    )

    data["price_pct_change_7"] = (
        grouped_price
        .pct_change(
            7,
            fill_method=None,
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(-2, 2)
        .astype("float32")
    )

    return data

def build_static_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build features that do not depend on past sales values.
    """
    data = add_calendar_features(data)
    data = add_cyclical_features(data)
    data = add_event_features(data)
    data = add_snap_features(data)
    data = add_price_features(data)

    return data

def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged sales features for each store-category time series.
    """
    data = data.copy()

    group_columns = [
        "store_id",
        "cat_id",
    ]

    data = data.sort_values(
        group_columns + ["date"]
    )

    grouped_sales = data.groupby(
        group_columns,
        observed=True,
    )["sales"]

    lag_days = [
        1,
        7,
        14,
        21,
        28,
        35,
        42,
        56,
    ]

    for lag in lag_days:
        data[f"lag_{lag}"] = (
            grouped_sales
            .shift(lag)
            .astype("float32")
        )

    return data
def add_rolling_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add leakage-safe rolling sales statistics.

    Rolling statistics are computed only from observations
    prior to the current day.
    """
    data = data.copy()

    group_columns = [
        "store_id",
        "cat_id",
    ]

    data = data.sort_values(
        group_columns + ["date"]
    )

    grouped_sales = data.groupby(
        group_columns,
        observed=True,
    )["sales"]

    shifted_sales = grouped_sales.shift(1)

    rolling_mean_windows = [
        7,
        14,
        28,
        56,
    ]

    for window in rolling_mean_windows:
        data[f"rolling_mean_{window}"] = (
            shifted_sales
            .groupby(
                [
                    data["store_id"],
                    data["cat_id"],
                ]
            )
            .transform(
                lambda s: s.rolling(
                    window,
                    min_periods=1,
                ).mean()
            )
            .astype("float32")
        )

    rolling_std_windows = [
        7,
        14,
        28,
    ]

    for window in rolling_std_windows:
        data[f"rolling_std_{window}"] = (
            shifted_sales
            .groupby(
                [
                    data["store_id"],
                    data["cat_id"],
                ]
            )
            .transform(
                lambda s: s.rolling(
                    window,
                    min_periods=2,
                ).std()
            )
            .astype("float32")
        )

    return data
def add_rolling_summary_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add short-term rolling median, minimum, and maximum sales features.
    """
    data = data.copy()

    group_columns = [
        "store_id",
        "cat_id",
    ]

    data = data.sort_values(
        group_columns + ["date"]
    )

    shifted_sales = (
        data
        .groupby(
            group_columns,
            observed=True,
        )["sales"]
        .shift(1)
    )

    grouped_shifted = shifted_sales.groupby(
        [
            data["store_id"],
            data["cat_id"],
        ]
    )

    data["rolling_median_7"] = (
        grouped_shifted
        .transform(
            lambda s: s.rolling(
                7,
                min_periods=1,
            ).median()
        )
        .astype("float32")
    )

    data["rolling_min_7"] = (
        grouped_shifted
        .transform(
            lambda s: s.rolling(
                7,
                min_periods=1,
            ).min()
        )
        .astype("float32")
    )

    data["rolling_max_7"] = (
        grouped_shifted
        .transform(
            lambda s: s.rolling(
                7,
                min_periods=1,
            ).max()
        )
        .astype("float32")
    )

    return data
def add_dynamic_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build sales-history features for forecasting.
    """
    data = add_lag_features(data)
    data = add_rolling_features(data)
    data = add_rolling_summary_features(data)

    return data

def prepare_model_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove rows that do not yet have enough sales history
    for the full forecasting feature set.
    """
    required_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_21",
        "lag_28",
        "lag_35",
        "lag_42",
        "lag_56",
    ]

    model_data = (
        data
        .dropna(subset=required_features)
        .reset_index(drop=True)
    )

    return model_data
def build_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete forecasting feature dataset.
    """
    data = build_static_features(data)
    data = add_dynamic_features(data)
    data = prepare_model_features(data)

    return data
