from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import load_m5_data


def test_load_m5_data_raises_error_when_files_are_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_m5_data(tmp_path)


def test_load_m5_data_reads_expected_files(tmp_path: Path):
    sales = pd.DataFrame({"id": ["item_1"], "d_1": [10]})
    calendar = pd.DataFrame({"date": ["2026-01-01"], "d": ["d_1"]})
    prices = pd.DataFrame(
        {
            "store_id": ["CA_1"],
            "item_id": ["item_1"],
            "wm_yr_wk": [1],
            "sell_price": [3.99],
        }
    )

    sales.to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    calendar.to_csv(tmp_path / "calendar.csv", index=False)
    prices.to_csv(tmp_path / "sell_prices.csv", index=False)

    loaded_sales, loaded_calendar, loaded_prices = load_m5_data(tmp_path)

    pd.testing.assert_frame_equal(loaded_sales, sales)
    pd.testing.assert_frame_equal(loaded_calendar, calendar)
    pd.testing.assert_frame_equal(loaded_prices, prices)