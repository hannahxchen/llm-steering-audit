"""Functions for loading evaluation results from saved JSON files."""

import os
import re
from pathlib import Path
from typing import Optional, List
import pandas as pd
from scipy import stats

from ..eval import load_eval_task


def load_steering_results(artifact_dir: Path, task_name: str, concept: Optional[str] = None) -> pd.DataFrame:
    """Load steering evaluation results for a task.

    Args:
        artifact_dir: Directory containing evaluation results.
        task_name: Name of the task.
        concept: Optional concept for group_type parameter (used by admissions task).

    Returns:
        DataFrame with columns: group, score, coeff.
    """
    task = load_eval_task(task_name)
    outputs = []

    eval_dir = artifact_dir / f'evaluation/{task_name}'
    if not eval_dir.exists():
        return pd.DataFrame()

    for filepath in eval_dir.rglob(f'{task_name}*.json'):
        # Skip baseline files
        if 'baseline' in filepath.name:
            continue

        coeff_str = filepath.name.split("_coeff=")[-1].replace(".json", "")
        try:
            coeff = float(coeff_str)
        except ValueError:
            continue

        if task_name == "admissions":
            result = task.compute_result_by_group(filepath, group_type=concept)
        else:
            result = task.compute_result_by_group(filepath)

        for group, score in result.items():
            outputs.append({
                "group": group,
                "score": score,
                "coeff": coeff
            })

    return pd.DataFrame.from_records(outputs)


def load_blackbox_results(artifact_dir: Path, task_name: str, concept: Optional[str] = None) -> pd.DataFrame:
    """Load black-box baseline evaluation results for a task.

    Args:
        artifact_dir: Directory containing evaluation results.
        task_name: Name of the task.
        concept: Optional concept for group_type parameter (used by admissions task).

    Returns:
        DataFrame with columns: group, score, explicit.
    """
    task = load_eval_task(task_name)
    outputs = []

    eval_dir = artifact_dir / f'evaluation/{task_name}'
    if not eval_dir.exists():
        return pd.DataFrame()

    for filepath in eval_dir.rglob(f'{task_name}*.json'):
        # Only process baseline files
        if re.search(r"_explicit-baseline.json$", filepath.name):
            is_explicit = True
        elif re.search(r"-baseline.json$", filepath.name):
            is_explicit = False
        else:
            continue

        if task_name == "admissions":
            result = task.compute_result_by_group(filepath, group_type=concept)
        else:
            result = task.compute_result_by_group(filepath)

        for group, score in result.items():
            outputs.append({
                "group": group,
                "score": score,
                "explicit": is_explicit
            })

    return pd.DataFrame.from_records(outputs)


def get_steering_evaluation_results(
    artifact_dir: Optional[Path] = None,
    concept: Optional[str] = None,
    train_dataset: Optional[str] = None,
    task_name: str = "judicial_guilt",
    model_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute white-box bias metrics (slopes) across multiple models.

    Args:
        artifact_dir: Root directory containing model results. If None, uses
            f"runs/{concept}-{train_dataset}".
        concept: Target concept (for default artifact_dir path).
        train_dataset: Training dataset name (for default artifact_dir path).
        task_name: Task to evaluate.
        model_list: List of model names to include. If None, uses all directories
            in artifact_dir.

    Returns:
        DataFrame with columns: model, task_name, slope, p.
    """
    if artifact_dir is None:
        if concept is None or train_dataset is None:
            raise ValueError("Must provide artifact_dir or both concept and train_dataset")
        artifact_dir = Path(f"runs/{concept}-{train_dataset}")

    if model_list is None:
        if not artifact_dir.exists():
            return pd.DataFrame()
        model_list = [d for d in os.listdir(artifact_dir) if (artifact_dir / d).is_dir()]

    results = []
    for model in model_list:
        model_dir = artifact_dir / model
        df = load_steering_results(model_dir, task_name=task_name, concept=concept)

        if df.empty:
            continue

        # For diversitymedqa, filter to neutral group
        if task_name.startswith("diversitymedqa"):
            df = df[df.group == "neutral"]

        if len(df) < 2:
            continue

        x = df.coeff.to_numpy()
        y = df["score"].to_numpy()
        slope, intercept, r, p, std_err = stats.linregress(x, y)

        results.append({
            "model": model,
            "task_name": task_name,
            "slope": slope,
            "p": p,
        })

    return pd.DataFrame.from_records(results)


def get_blackbox_evaluation_results(
    concept: str,
    artifact_dir: Path = Path("runs/baseline"),
    task_name: str = "judicial_guilt",
    model_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute black-box bias metrics (group disparities) across multiple models.

    Args:
        concept: Target concept ("gender" or "race") for computing group disparity.
        artifact_dir: Root directory containing baseline results.
        task_name: Task to evaluate.
        model_list: List of model names to include. If None, uses all directories
            in artifact_dir.

    Returns:
        DataFrame with columns: model, task_name, explicit, group_diff.
    """
    if model_list is None:
        if not artifact_dir.exists():
            return pd.DataFrame()
        model_list = [d for d in os.listdir(artifact_dir) if (artifact_dir / d).is_dir()]

    results = []
    for model in model_list:
        model_dir = artifact_dir / model
        df = load_blackbox_results(model_dir, task_name=task_name, concept=concept)

        if df.empty:
            continue

        for is_explicit in [True, False]:
            explicit_df = df[df.explicit == is_explicit]
            if explicit_df.empty:
                continue

            # Compute group disparity
            group_scores = explicit_df.set_index("group")["score"].to_dict()

            if concept == "race":
                if "black" in group_scores and "white" in group_scores:
                    group_diff = group_scores["black"] - group_scores["white"]
                else:
                    continue
            elif concept == "gender":
                if "female" in group_scores and "male" in group_scores:
                    group_diff = group_scores["female"] - group_scores["male"]
                else:
                    continue
            else:
                continue

            results.append({
                "model": model,
                "task_name": task_name,
                "explicit": is_explicit,
                "group_diff": group_diff
            })

    return pd.DataFrame.from_records(results)
