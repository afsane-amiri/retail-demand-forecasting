import pandas as pd


TARGET_COLUMN = "sales"

# Numeric forecasting features that can be used directly by ML models.
NUMERIC_FEATURES = [
    # Calendar
    "day",
    "dayofweek",
    "week",
    "month",
    "quarter",
    "dayofyear",
    "is_weekend",
    "time_idx",

    # Cyclical calendar
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",

    # Events and SNAP
    "is_event_day",
    "has_two_events",
    "snap",

    # Price
    "avg_sell_price",
    "price_change_1",
    "price_change_7",
    "price_mean_28",
    "price_vs_mean_28",
    "price_pct_change_7",

    # Demand lags
    "lag_28",
    "lag_35",
    "lag_42",
    "lag_56",

]

CATEGORICAL_FEATURES = [
    "store_id",
    "state_id",
    "cat_id",
]


def prepare_ml_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate model predictors from the forecasting target.
    """
    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    missing_columns = [
        column
        for column in feature_columns + [TARGET_COLUMN]
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required model columns are missing: {missing_columns}"
        )

    X = data[feature_columns].copy()
    y = data[TARGET_COLUMN].copy()

    return X, y