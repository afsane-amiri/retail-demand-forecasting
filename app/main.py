from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------------------
# Project and model configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "production_model"
    / "model.joblib"
)


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
    """Load the exported production model."""

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Production model was not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# Load the model once when the API starts.
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the production model once when the API starts."""

    global model

    model = load_production_model()

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