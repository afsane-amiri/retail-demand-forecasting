import pandas as pd


def validate_sales_data(sales: pd.DataFrame) -> None:
    """
    Validate the raw M5 sales dataset.

    Raises
    ------
    ValueError
        If required columns are missing or invalid values are found.
    """
    required_columns = {
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    }

    missing_columns = required_columns - set(sales.columns)

    if missing_columns:
        raise ValueError(
            f"Sales data is missing required columns: {sorted(missing_columns)}"
        )

    if sales["id"].duplicated().any():
        raise ValueError("Sales data contains duplicate series IDs.")

    day_columns = [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]

    if not day_columns:
        raise ValueError("Sales data contains no daily sales columns.")

    if sales[day_columns].isna().any().any():
        raise ValueError("Sales data contains missing sales values.")

    if (sales[day_columns] < 0).any().any():
        raise ValueError("Sales data contains negative sales values.")


def validate_calendar_data(calendar: pd.DataFrame) -> None:
    """
    Validate the raw M5 calendar dataset.
    """
    required_columns = {
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "d",
    }

    missing_columns = required_columns - set(calendar.columns)

    if missing_columns:
        raise ValueError(
            f"Calendar data is missing required columns: {sorted(missing_columns)}"
        )

    if calendar["d"].duplicated().any():
        raise ValueError("Calendar data contains duplicate day identifiers.")

    if calendar["date"].isna().any():
        raise ValueError("Calendar data contains missing dates.")

    parsed_dates = pd.to_datetime(calendar["date"], errors="coerce")

    if parsed_dates.isna().any():
        raise ValueError("Calendar data contains invalid date values.")


def validate_price_data(prices: pd.DataFrame) -> None:
    """
    Validate the raw M5 selling-price dataset.
    """
    required_columns = {
        "store_id",
        "item_id",
        "wm_yr_wk",
        "sell_price",
    }

    missing_columns = required_columns - set(prices.columns)

    if missing_columns:
        raise ValueError(
            f"Price data is missing required columns: {sorted(missing_columns)}"
        )

    if prices["sell_price"].isna().any():
        raise ValueError("Price data contains missing selling prices.")

    if (prices["sell_price"] <= 0).any():
        raise ValueError("Price data contains non-positive selling prices.")

    duplicate_keys = prices.duplicated(
        subset=["store_id", "item_id", "wm_yr_wk"]
    )

    if duplicate_keys.any():
        raise ValueError(
            "Price data contains duplicate store-item-week records."
        )


def validate_m5_data(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """
    Validate all three raw M5 datasets.
    """
    validate_sales_data(sales)
    validate_calendar_data(calendar)
    validate_price_data(prices)