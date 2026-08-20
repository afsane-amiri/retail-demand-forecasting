from pathlib import Path

import pandas as pd


def load_m5_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the raw M5 Forecasting datasets.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the raw M5 CSV files.

    Returns
    -------
    sales : pd.DataFrame
        Historical unit sales data.

    calendar : pd.DataFrame
        Calendar, event, SNAP, and date information.

    prices : pd.DataFrame
        Weekly selling prices by item and store.
    """
    data_dir = Path(data_dir)

    sales_path = data_dir / "sales_train_evaluation.csv"
    calendar_path = data_dir / "calendar.csv"
    prices_path = data_dir / "sell_prices.csv"

    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)

    return sales, calendar, prices