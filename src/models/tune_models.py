import numpy as np
import pandas as pd
from itertools import product
import random

from src.evaluation.metrics import (
    mae,
    rmse,
    wape,
    bias_pct,
)
from src.models.model_data import prepare_ml_data


def create_time_series_cv_folds(
    data: pd.DataFrame,
    n_splits: int = 4,
    horizon_days: int = 28,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create expanding-window cross-validation folds using complete dates.

    Each validation fold contains `horizon_days` consecutive calendar
    days across all forecasting series. Training contains all observations
    strictly before the validation period.

    The most recent folds are used so that cross-validation reflects
    relatively recent forecasting conditions.
    """
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])

    unique_dates = (
        data["date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    required_validation_days = n_splits * horizon_days

    if len(unique_dates) <= required_validation_days:
        raise ValueError(
            "Not enough historical dates to create the requested "
            "cross-validation folds."
        )

    folds = []

    # Reserve consecutive validation blocks working backward
    # from the end of the supplied training history.
    first_validation_start = (
        len(unique_dates) - required_validation_days
    )

    for fold_number in range(n_splits):
        validation_start_idx = (
            first_validation_start
            + fold_number * horizon_days
        )

        validation_end_idx = (
            validation_start_idx
            + horizon_days
        )

        validation_dates = unique_dates.iloc[
            validation_start_idx:validation_end_idx
        ]

        validation_start = validation_dates.iloc[0]
        validation_end = validation_dates.iloc[-1]

        train_fold = data[
            data["date"] < validation_start
        ].copy()

        validation_fold = data[
            (data["date"] >= validation_start)
            & (data["date"] <= validation_end)
        ].copy()

        folds.append(
            (
                train_fold,
                validation_fold,
            )
        )

    return folds

def evaluate_model_cv(
    model_builder,
    model_params: dict,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> dict:
    """
    Evaluate one model configuration across time-series CV folds.

    A new model is created for every fold to ensure that information
    learned from one validation period does not carry into another.
    """
    fold_results = []

    for fold_number, (train_fold, validation_fold) in enumerate(
        folds,
        start=1,
    ):
        X_train, y_train = prepare_ml_data(train_fold)
        X_validation, y_validation = prepare_ml_data(
            validation_fold
        )

        model = model_builder(**model_params)

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        fold_results.append(
            {
                "fold": fold_number,
                "mae": mae(y_validation, predictions),
                "rmse": rmse(y_validation, predictions),
                "wape": wape(y_validation, predictions),
                "bias": bias_pct(y_validation, predictions),
            }
        )

    results = pd.DataFrame(fold_results)

    return {
        "mean_mae": results["mae"].mean(),
        "mean_rmse": results["rmse"].mean(),
        "mean_wape": results["wape"].mean(),
        "mean_bias": results["bias"].mean(),
        "std_wape": results["wape"].std(),
        "fold_results": results,
    }
def search_hyperparameters(
    model_builder,
    param_grid: dict,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Evaluate all parameter combinations using time-series CV.

    Configurations are ranked by mean WAPE across validation folds.
    Progress is printed after each configuration.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    combinations = list(product(*param_values))
    total_combinations = len(combinations)

    results = []

    for config_number, values in enumerate(
        combinations,
        start=1,
    ):
        params = dict(zip(param_names, values))

        print(
            f"Configuration "
            f"{config_number}/{total_combinations}: "
            f"{params}"
        )

        cv_results = evaluate_model_cv(
            model_builder=model_builder,
            model_params=params,
            folds=folds,
        )

        results.append(
            {
                **params,
                "mean_wape": cv_results["mean_wape"],
                "std_wape": cv_results["std_wape"],
                "mean_mae": cv_results["mean_mae"],
                "mean_rmse": cv_results["mean_rmse"],
                "mean_bias": cv_results["mean_bias"],
            }
        )

        print(
            f"  Mean WAPE: "
            f"{cv_results['mean_wape']:.2f}%"
        )

    return (
        pd.DataFrame(results)
        .sort_values("mean_wape")
        .reset_index(drop=True)
    )
def randomized_hyperparameter_search(
    model_builder,
    param_grid: dict,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    n_iter: int = 12,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Randomly sample parameter configurations and evaluate each
    using time-series cross-validation.

    Configurations are ranked by mean WAPE across validation folds.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    all_combinations = list(product(*param_values))

    if n_iter > len(all_combinations):
        raise ValueError(
            "n_iter cannot exceed the total number of "
            "parameter combinations."
        )

    rng = random.Random(random_state)

    sampled_combinations = rng.sample(
        all_combinations,
        n_iter,
    )

    results = []

    for config_number, values in enumerate(
        sampled_combinations,
        start=1,
    ):
        params = dict(zip(param_names, values))

        print(
            f"Configuration {config_number}/{n_iter}: "
            f"{params}"
        )

        cv_results = evaluate_model_cv(
            model_builder=model_builder,
            model_params=params,
            folds=folds,
        )

        results.append(
            {
                **params,
                "mean_wape": cv_results["mean_wape"],
                "std_wape": cv_results["std_wape"],
                "mean_mae": cv_results["mean_mae"],
                "mean_rmse": cv_results["mean_rmse"],
                "mean_bias": cv_results["mean_bias"],
            }
        )

        print(
            f"  Mean WAPE: "
            f"{cv_results['mean_wape']:.2f}%"
        )

    return (
        pd.DataFrame(results)
        .sort_values("mean_wape")
        .reset_index(drop=True)
    )