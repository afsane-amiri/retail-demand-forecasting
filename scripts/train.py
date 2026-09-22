from pathlib import Path
import sys

import mlflow
import mlflow.sklearn
import pandas as pd


# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.load_data import load_m5_data
from src.data.preprocess import preprocess_m5_data
from src.data.split_data import (
    time_series_split,
    validate_time_series_split,
)
from src.features.build_features import build_features
from src.models.model_data import prepare_ml_data
from src.models.train_random_forest import (
    build_random_forest_pipeline,
)
from src.evaluation.metrics import (
    mae,
    rmse,
    wape,
    bias_pct,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "raw"

EXPERIMENT_NAME = "retail-demand-forecasting"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

MLFLOW_TRACKING_URI = MLFLOW_DB.as_uri().replace(
    "file:///",
    "sqlite:///",
)

BEST_RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": 42,
}


def main() -> None:
    """Train and evaluate the final forecasting model."""

    # -----------------------------------------------------------------
    # Load and prepare data
    # -----------------------------------------------------------------

    print("Loading raw M5 data...")

    sales, calendar, prices = load_m5_data(
        DATA_DIR
    )

    print("Preprocessing data...")

    data = preprocess_m5_data(
        sales,
        calendar,
        prices,
    )

    print("Building forecasting features...")

    model_data = build_features(data)

    # -----------------------------------------------------------------
    # Time-based split
    # -----------------------------------------------------------------

    train, validation, test = time_series_split(
        model_data,
        validation_days=28,
        test_days=28,
    )

    validate_time_series_split(
        train,
        validation,
        test,
    )

    # Model selection is already complete, so validation history can
    # now be included when fitting the locked final model.
    train_validation = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    X_train, y_train = prepare_ml_data(
        train_validation
    )

    X_test, y_test = prepare_ml_data(test)

    print(
        f"Training period: "
        f"{train_validation['date'].min().date()} "
        f"to {train_validation['date'].max().date()}"
    )

    print(
        f"Test period: "
        f"{test['date'].min().date()} "
        f"to {test['date'].max().date()}"
    )

    # -----------------------------------------------------------------
    # MLflow
    # -----------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name="final_random_forest"
    ):
        mlflow.log_params(BEST_RF_PARAMS)

        mlflow.log_param(
            "forecast_horizon_days",
            28,
        )

        mlflow.log_param(
            "training_rows",
            len(X_train),
        )

        mlflow.log_param(
            "test_rows",
            len(X_test),
        )

        # -------------------------------------------------------------
        # Train
        # -------------------------------------------------------------

        print("Training final Random Forest...")

        model = build_random_forest_pipeline(
            **BEST_RF_PARAMS
        )

        model.fit(
            X_train,
            y_train,
        )

        # -------------------------------------------------------------
        # Evaluate
        # -------------------------------------------------------------

        predictions = model.predict(X_test)

        test_mae = mae(
            y_test,
            predictions,
        )

        test_rmse = rmse(
            y_test,
            predictions,
        )

        test_wape = wape(
            y_test,
            predictions,
        )

        test_bias = bias_pct(
            y_test,
            predictions,
        )

        mlflow.log_metrics(
            {
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_wape": test_wape,
                "test_bias_pct": test_bias,
            }
        )

        # -------------------------------------------------------------
        # Save model artifact
        # -------------------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="skops",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
            ],
        )

        print("\nFinal test metrics")
        print("------------------")
        print(f"MAE:  {test_mae:.2f}")
        print(f"RMSE: {test_rmse:.2f}")
        print(f"WAPE: {test_wape:.2f}%")
        print(f"Bias: {test_bias:.2f}%")

        print(
            "\nMLflow run ID:",
            mlflow.active_run().info.run_id,
        )


if __name__ == "__main__":
    main()