from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn


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

EXPORT_DIR = PROJECT_ROOT / "models" / "production_model"


def main() -> None:
    """Export the latest successful production model from MLflow."""

    # Connect to the local MLflow tracking database.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Find the forecasting experiment.
    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found."
        )

    # Find the most recent successful final Random Forest run.
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

    print(
        f"Loading model from MLflow run: {run_id}"
    )

    # Load the fitted sklearn pipeline from MLflow.
    model = mlflow.sklearn.load_model(
        model_uri
    )

    # Create the deployment-model directory.
    EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_path = EXPORT_DIR / "model.joblib"

    # Save the exact fitted pipeline in a more compact format.
    joblib.dump(
        model,
        export_path,
        compress=3,
    )

    size_mb = export_path.stat().st_size / (1024 ** 2)

    print(
        f"Production model exported to: {export_path}"
    )

    print(
        f"Model size: {size_mb:.2f} MB"
    )


if __name__ == "__main__":
    main()