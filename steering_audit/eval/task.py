from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Callable, Self, Any, Dict
import pandas as pd
from ..config import EVAL_DATA_DIR
from ..utils import save_to_json_file

class Task(ABC):
    """Abstract base class for evaluation tasks.

    Each task defines how to load its data, prepare inputs for the model,
    save outputs, and compute aggregated results by group.

    Attributes:
        task_name: Unique identifier for the task.
        explicit: Whether to include explicit protected attributes in prompts.
        output_labels: List of possible output labels for token-probability tasks.
        max_new_tokens: Maximum tokens to generate for open-ended generation tasks.
        dataset: Loaded task data (set during init via _load_data).
    """
    def __init__(
        self, task_name: str,
        output_labels: List[str] = None,
        max_new_tokens: int = 15,
        explicit: bool = False,
    ):
        self.task_name = task_name
        self.explicit = explicit
        self.eval_data_dir = EVAL_DATA_DIR
        self.dataset = self._load_data()
        self.output_labels = output_labels
        self.max_new_tokens = max_new_tokens

    @classmethod
    @abstractmethod
    def _load_data(cls) -> Self:
        """Load the task dataset.

        Returns:
            List of data items (dicts) for the task.
        """
        pass

    @abstractmethod
    def prepare_inputs(self, chat_template_func: Callable, *kwargs) -> List[str]:
        """Prepare formatted input prompts for the model.

        Args:
            chat_template_func: Function to apply chat template formatting.
            *kwargs: Additional arguments for prompt preparation.

        Returns:
            List of formatted prompt strings.
        """
        pass

    @abstractmethod
    def save_outputs(self, outputs: Any, save_filepath: Path):
        """Save model outputs to disk.

        Args:
            outputs: Raw model outputs (token probs or generated text).
            save_filepath: Path to save results to.
        """
        pass

    @abstractmethod
    def load_and_process_result(self, output_filepath: Path) -> pd.DataFrame:
        """Load and process saved results into a DataFrame.

        Args:
            output_filepath: Path to the saved results file.

        Returns:
            DataFrame containing processed results.
        """
        pass

    @abstractmethod
    def compute_result_by_group(self, output_filepath: Path) -> Dict:
        """Compute aggregated metrics by protected group.

        Args:
            output_filepath: Path to the saved results file.

        Returns:
            Dict mapping group names to metric values.
        """
        pass


class TokenProbabilityTaskMixin:
    """Mixin for tasks that output token probabilities.

    Provides a default implementation of save_outputs for tasks that
    compute next-token probabilities over a set of output labels.
    """

    def save_outputs(self, outputs, save_filepath: Path):
        """Save token probability outputs.

        Args:
            outputs: Array of token probabilities [n_examples, n_labels].
            save_filepath: Path to save results to.
        """
        results = []
        for item, probs in zip(self.dataset, outputs):
            result = dict(item)  # Copy item data
            result["output_probs"] = probs.tolist()
            results.append(result)
        save_to_json_file(results, save_filepath)
