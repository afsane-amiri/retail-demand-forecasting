from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient

import app.main as main


mock_model = MagicMock()
mock_model.predict.return_value = np.array([125.5])

main.model = mock_model
main.model_run_id = "test-run-id"


client = TestClient(main.app)


SAMPLE_REQUEST = {
    "day": 25,
    "dayofweek": 0,
    "week": 17,
    "month": 4,
    "quarter": 2,
    "dayofyear": 116,
    "is_weekend": 0,
    "time_idx": 1913,

    "dow_sin": 0.0,
    "dow_cos": 1.0,
    "month_sin": 0.866,
    "month_cos": -0.5,
    "doy_sin": 0.91,
    "doy_cos": -0.41,

    "is_event_day": 0,
    "has_two_events": 0,
    "snap": 0,

    "avg_sell_price": 3.5,
    "price_change_1": 0.0,
    "price_change_7": 0.1,
    "price_mean_28": 3.4,
    "price_vs_mean_28": 1.03,
    "price_pct_change_7": 0.02,

    "lag_28": 1200.0,
    "lag_35": 1180.0,
    "lag_42": 1210.0,
    "lag_56": 1190.0,

    "store_id": "CA_1",
    "state_id": "CA",
    "cat_id": "FOODS",
}


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Retail Demand Forecasting API",
        "status": "running",
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == "random_forest"
    assert response.json()["model_run_id"] == "test-run-id"


def test_predict():
    response = client.post(
        "/predict",
        json=SAMPLE_REQUEST,
    )

    assert response.status_code == 200

    assert response.json() == {
        "predicted_sales": 125.5
    }


def test_predict_rejects_missing_features():
    invalid_request = SAMPLE_REQUEST.copy()

    invalid_request.pop("lag_28")

    response = client.post(
        "/predict",
        json=invalid_request,
    )

    assert response.status_code == 422