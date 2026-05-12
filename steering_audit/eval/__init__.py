from .evaluator import Evaluator
from .judicial import Judicial
from .admissions import Admissions
from .south_german import SouthGerman
from .diversitymedqa import DiversityMedQA
from ..constants import EVAL_TASKS


def load_eval_task(task_name: str, explicit: bool = False):
    """Load an evaluation task by name.

    Args:
        task_name: Name of the task to load (must be in EVAL_TASKS).
        explicit: Whether to use explicit protected attributes in prompts.

    Returns:
        Task instance for the specified task.

    Raises:
        AssertionError: If task_name is not a valid task.
        ValueError: If task type is unknown.
    """
    assert task_name in EVAL_TASKS, f"Unknown task: {task_name}. Must be one of {EVAL_TASKS}"
    if task_name.startswith("judicial_"):
        return Judicial(task_name, explicit=explicit)
    elif task_name == "admissions":
        return Admissions(explicit=explicit)
    elif task_name.startswith("south_german"):
        return SouthGerman(explicit=explicit, task_name=task_name)
    elif task_name.startswith("diversitymedqa"):
        return DiversityMedQA(task_name)
    else:
        raise ValueError(f"Unknown task name: {task_name}")
    