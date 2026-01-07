import os
from pathlib import Path
from dataclasses import dataclass
from dataclass_wizard import YAMLWizard
from typing import Self

EVAL_DATA_DIR = Path.cwd() / "steering_audit/data/eval_data"

@dataclass
class SteeringConfig:
    layer_id: int # Layer to intervene
    coeff: float # Steering coefficient
    min_coeff: float
    max_coeff: float
    increment: float

@dataclass
class EvalConfig:
    batch_size: int
    generation_batch_size: int
    max_new_tokens: int
    num_return_sequences: int
    do_sample: bool = True 
    temperature: float = None
    top_p: float = None


@dataclass
class Config(YAMLWizard):
    model_name: str
    concept: str # Target concept
    dataset: str # Dataset for extracting the concept
    n_train: int # Training size; Use all samples if -1
    n_val: int # Validation size
    method: str # Vector extraction method
    threshold: float # Threshold score for labeling
    weighted_sample: bool # Apply weighted sampling to select training prompts
    filter_layer_pct: float # Filter last N% layers
    save_dir: str # Directory where results are saved
    seed: int # Random seed

    def __post_init__(self):
        self.model_alias = os.path.basename(self.model_name)
        if self.save_dir is None:
            self.save_dir = f"runs/{self.concept}-{self.dataset}/{self.model_alias}"
    
    def artifact_path(self) -> Path:
        return Path().absolute() / self.save_dir

    def save(self):
        os.makedirs(self.artifact_path(), exist_ok=True)
        self.to_yaml_file(self.artifact_path() / 'config.yaml')
    
    def load(filepath: str) -> Self:
        try:
            return Config.from_yaml_file(filepath)
        
        except FileNotFoundError:
            return None
