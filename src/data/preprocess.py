import pandas as pd


SERIES_COLUMNS = [
    "store_id",
    "state_id",
    "cat_id",
]


def aggregate_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate item-level M5 sales to the store-category level.

    Aggregation is performed while the sales dataset is still in
    wide format to avoid creating the much larger item-level
    long-format table.

    Parameters
    ----------
    sales : pd.DataFrame
        Raw M5 sales data in wide format.

    Returns
    -------
    pd.DataFrame
        Store-category sales data in wide format.
    """
    day_columns = [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]

    aggregated_sales = (
        sales
        .groupby(
            SERIES_COLUMNS,
            observed=True,
        )[day_columns]
        .sum()
        .reset_index()
    )

    return aggregated_sales


def reshape_sales_to_long(
    aggregated_sales: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert store-category sales from wide to long format.

    Parameters
    ----------
    aggregated_sales : pd.DataFrame
        Store-category sales in wide format.

    Returns
    -------
    pd.DataFrame
        Daily store-category sales in long format.
    """
    day_columns = [
        column
        for column in aggregated_sales.columns
        if column.startswith("d_")
    ]

    sales_long = aggregated_sales.melt(
        id_vars=SERIES_COLUMNS,
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )

    sales_long["sales"] = pd.to_numeric(
        sales_long["sales"],
        downcast="integer",
    )

    return sales_long
def aggregate_prices(
    prices: pd.DataFrame,
    sales: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate item-level weekly prices to the store-category level.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw M5 weekly selling-price data.
    sales : pd.DataFrame
        Raw M5 sales data, used to map items to categories.

    Returns
    -------
    pd.DataFrame
        Average weekly selling price for each store-category combination.
    """
    item_categories = (
        sales[
            [
                "store_id",
                "item_id",
                "cat_id",
            ]
        ]
        .drop_duplicates()
    )

    prices_with_category = prices.merge(
        item_categories,
        on=["store_id", "item_id"],
        how="left",
        validate="many_to_one",
    )

    aggregated_prices = (
        prices_with_category
        .groupby(
            [
                "store_id",
                "cat_id",
                "wm_yr_wk",
            ],
            observed=True,
        )["sell_price"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "sell_price": "avg_sell_price",
            }
        )
    )

    return aggregated_prices
def merge_calendar(
    sales_long: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add calendar information to daily store-category sales.

    Parameters
    ----------
    sales_long : pd.DataFrame
        Daily store-category sales in long format.
    calendar : pd.DataFrame
        Raw M5 calendar data.

    Returns
    -------
    pd.DataFrame
        Sales data enriched with calendar information.
    """
    calendar = calendar.copy()

    calendar["date"] = pd.to_datetime(
        calendar["date"]
    )

    data = sales_long.merge(
        calendar,
        on="d",
        how="left",
        validate="many_to_one",
    )

    return data
def merge_prices(
    data: pd.DataFrame,
    aggregated_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add weekly store-category price information.

    Parameters
    ----------
    data : pd.DataFrame
        Daily sales data containing calendar information.
    aggregated_prices : pd.DataFrame
        Weekly store-category price data.

    Returns
    -------
    pd.DataFrame
        Sales and calendar data enriched with price information.
    """
    data = data.merge(
        aggregated_prices,
        on=[
            "store_id",
            "cat_id",
            "wm_yr_wk",
        ],
        how="left",
        validate="many_to_one",
    )

    return data

def preprocess_m5_data(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the base store-category forecasting dataset.

    The pipeline:
    1. Aggregates item-level sales to store-category level.
    2. Reshapes sales from wide to long format.
    3. Aggregates item prices to store-category-week level.
    4. Adds calendar information.
    5. Adds weekly price information.

    Returns
    -------
    pd.DataFrame
        Base dataset ready for feature engineering.
    """
    aggregated_sales = aggregate_sales(sales)

    sales_long = reshape_sales_to_long(
        aggregated_sales
    )

    aggregated_prices = aggregate_prices(
        prices,
        sales,
    )

    data = merge_calendar(
        sales_long,
        calendar,
    )

    data = merge_prices(
        data,
        aggregated_prices,
    )

    return data