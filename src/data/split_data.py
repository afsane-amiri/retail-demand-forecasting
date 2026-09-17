import pandas as pd


def time_series_split(
    data: pd.DataFrame,
    validation_days: int = 28,
    test_days: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split forecasting data chronologically into train,
    validation, and test sets.

    The final `test_days` are reserved for final evaluation,
    while the preceding `validation_days` are used for model
    selection and tuning.
    """
    data = data.copy()

    data["date"] = pd.to_datetime(data["date"])

    max_date = data["date"].max()

    test_start = (
        max_date
        - pd.Timedelta(days=test_days - 1)
    )

    validation_end = (
        test_start
        - pd.Timedelta(days=1)
    )

    validation_start = (
        validation_end
        - pd.Timedelta(days=validation_days - 1)
    )

    train = data[
        data["date"] < validation_start
    ].copy()

    validation = data[
        (data["date"] >= validation_start)
        & (data["date"] <= validation_end)
    ].copy()

    test = data[
        data["date"] >= test_start
    ].copy()

    return train, validation, test

def validate_time_series_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Validate chronological train, validation, and test splits.
    """
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Train, validation, and test sets must all contain data."
        )

    if train["date"].max() >= validation["date"].min():
        raise ValueError(
            "Train and validation periods overlap."
        )

    if validation["date"].max() >= test["date"].min():
        raise ValueError(
            "Validation and test periods overlap."
        )
    
    series_columns = [
        "store_id",
        "cat_id",
    ]

    train_series = set(
        train[series_columns]
        .itertuples(index=False, name=None)
    )

    validation_series = set(
        validation[series_columns]
        .itertuples(index=False, name=None)
    )

    test_series = set(
        test[series_columns]
        .itertuples(index=False, name=None)
    )

    if not (
        train_series
        == validation_series
        == test_series
    ):
        raise ValueError(
            "Series coverage differs across the data splits."
        )