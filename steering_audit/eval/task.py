from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Callable, Self, Any, Dict
import pandas as pd
from ..config import EVAL_DATA_DIR

class Task(ABC):
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
        pass
    
    @abstractmethod
    def prepare_inputs(self, chat_template_func: Callable, *kwargs) -> List[str]:
        pass
    
    @abstractmethod
    def save_outputs(self, outputs: Any, save_filepath: Path):
        pass

    @abstractmethod
    def load_and_process_result(self, output_filepath: Path) -> pd.DataFrame:
        pass

    @abstractmethod
    def compute_result_by_group(self, output_filepath: Path) -> Dict:
        pass
