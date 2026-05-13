"""Metric computation functions for bias analysis."""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_group_disparity(group_scores: Dict[str, float], concept: str) -> Optional[float]:
    """Compute disparity between protected groups.

    Args:
        group_scores: Dict mapping group names to scores.
        concept: Target concept ("gender" or "race").

    Returns:
        Group disparity (diff between disadvantaged and advantaged groups),
        or None if required groups not present.
    """
    if concept == "race":
        if "black" in group_scores and "white" in group_scores:
            return group_scores["black"] - group_scores["white"]
    elif concept == "gender":
        if "female" in group_scores and "male" in group_scores:
            return group_scores["female"] - group_scores["male"]
    return None


def compute_bias_metrics(
    df: pd.DataFrame,
    metric_type: str = "whitebox"
) -> Dict[str, float]:
    """Compute bias metrics from evaluation results.

    For white-box (steering): computes linear regression slope of score vs coefficient.
    For black-box: computes average group disparity.

    Args:
        df: DataFrame with evaluation results.
            For whitebox: columns should include 'coeff', 'score'
            For blackbox: columns should include 'group', 'score'
        metric_type: Either "whitebox" or "blackbox".

    Returns:
        Dict with computed metrics.

    Raises:
        ValueError: If metric_type is invalid or required columns missing.
    """
    if metric_type == "whitebox":
        required = {"coeff", "score"}
        if not required.issubset(df.columns):
            raise ValueError(f"White-box metrics require columns: {required}")

        x = df.coeff.to_numpy()
        y = df["score"].to_numpy()

        if len(x) < 2:
            return {"slope": np.nan, "intercept": np.nan, "r": np.nan, "p": np.nan}

        slope, intercept, r, p, std_err = stats.linregress(x, y)

        return {
            "slope": slope,
            "intercept": intercept,
            "r": r,
            "p": p,
            "std_err": std_err,
        }

    elif metric_type == "blackbox":
        required = {"group", "score"}
        if not required.issubset(df.columns):
            raise ValueError(f"Black-box metrics require columns: {required}")

        # Return group means
        return df.groupby("group")["score"].mean().to_dict()

    else:
        raise ValueError(f"Unknown metric_type: {metric_type}. Use 'whitebox' or 'blackbox'.")


def compute_steering_sensitivity(
    coeffs: np.ndarray,
    scores: np.ndarray
) -> Dict[str, float]:
    """Compute sensitivity metric from steering coefficient sweep.

    This is the core white-box metric: how much does the output change
    as we steer along the concept vector.

    Args:
        coeffs: Array of steering coefficients used.
        scores: Array of corresponding output scores.

    Returns:
        Dict with slope, p-value, and other regression statistics.
    """
    if len(coeffs) < 2:
        return {"slope": np.nan, "p": np.nan, "r": np.nan}

    slope, intercept, r, p, std_err = stats.linregress(coeffs, scores)

    return {
        "slope": slope,
        "intercept": intercept,
        "r": r,
        "r_squared": r ** 2,
        "p": p,
        "std_err": std_err,
    }
