from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.model_data import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def build_random_forest_pipeline(
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    max_features: str | float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a preprocessing and Random Forest forecasting pipeline.

    Numeric features pass through unchanged, while categorical
    features are one-hot encoded before model training.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )