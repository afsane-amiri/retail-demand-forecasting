from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------------------
# Project and MLflow configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

MLFLOW_TRACKING_URI = MLFLOW_DB.as_uri().replace(
    "file:///",
    "sqlite:///",
)

EXPERIMENT_NAME = "retail-demand-forecasting"
MODEL_RUN_NAME = "final_random_forest"



# ---------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Model-ready features for one store-category forecast."""

    # Calendar features
    day: int
    dayofweek: int
    week: int
    month: int
    quarter: int
    dayofyear: int
    is_weekend: int
    time_idx: int

    # Cyclical calendar features
    dow_sin: float
    dow_cos: float
    month_sin: float
    month_cos: float
    doy_sin: float
    doy_cos: float

    # Event and SNAP features
    is_event_day: int
    has_two_events: int
    snap: int

    # Price features
    avg_sell_price: float
    price_change_1: float
    price_change_7: float
    price_mean_28: float
    price_vs_mean_28: float
    price_pct_change_7: float

    # Horizon-safe demand lag features
    lag_28: float
    lag_35: float
    lag_42: float
    lag_56: float

    # Categorical features
    store_id: str
    state_id: str
    cat_id: str


class PredictionResponse(BaseModel):
    """Forecast returned by the model."""

    predicted_sales: float


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_production_model():
    """
    Load the most recent successful final Random Forest run
    from the MLflow experiment.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            f"AND tags.mlflow.runName = '{MODEL_RUN_NAME}'"
        ),
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError(
            "No successful final Random Forest run was found."
        )

    run_id = runs.iloc[0]["run_id"]

    model_uri = f"runs:/{run_id}/model"

    model = mlflow.sklearn.load_model(
        model_uri
    )

    return model, run_id


# Load the model once when the API starts.
model = None
model_run_id = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the production model once when the API starts."""

    global model, model_run_id

    model, model_run_id = load_production_model()

    yield

# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "API for serving forecasts from the retail "
        "demand forecasting model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "message": "Retail Demand Forecasting API",
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Return API and model status."""

    return {
        "status": "healthy",
        "model": "random_forest",
        "model_run_id": model_run_id,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Generate a sales forecast from model-ready features.
    """

    input_data = pd.DataFrame(
        [request.model_dump()]
    )

    prediction = model.predict(
        input_data
    )[0]

    # Unit demand cannot be negative.
    predicted_sales = max(
        0.0,
        float(prediction),
    )

    return PredictionResponse(
        predicted_sales=predicted_sales
    )