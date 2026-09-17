import numpy as np


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate mean absolute error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(
        np.mean(
            np.abs(y_true - y_pred)
        )
    )


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate root mean squared error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def wape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate weighted absolute percentage error.

    Returned as a percentage.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.sum(
        np.abs(y_true)
    )

    if denominator == 0:
        raise ValueError(
            "WAPE is undefined when total actual demand is zero."
        )

    return float(
        np.sum(
            np.abs(y_true - y_pred)
        )
        / denominator
        * 100
    )


def bias_pct(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate forecast bias as a percentage of actual demand.

    Positive values indicate overforecasting.
    Negative values indicate underforecasting.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.sum(y_true)

    if denominator == 0:
        raise ValueError(
            "Bias percentage is undefined when total actual demand is zero."
        )

    return float(
        np.sum(y_pred - y_true)
        / denominator
        * 100
    )