import os, logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Callable, Self
import pandas as pd
import numpy as np
import torch.nn.functional as F

from ..data import load_datasplit, load_target_words
from ..utils import PromptIterator, save_to_json_file
from .model import ModelBase
from .steering_utils import get_target_token_ids

concept_labels = {
    "gender": ("F", "M"), # (Female, Male)
    "race": ("B", "W") # (Black, White)
}


@dataclass
class Dataset:    
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    threshold: float
    pos_tokens: List[str]
    neg_tokens: List[str]

    @classmethod
    def load(cls, concept: str, dataset: str, n_val: int = None, threshold: float = 0, cached_dir: Path = None) -> Self:
        assert threshold >= 0
        train_data = load_datasplit(dataset, split="train", sample_size=-1, cached_dir=cached_dir)
        val_data = load_datasplit(dataset, split="val", sample_size=n_val, cached_dir=cached_dir)
        target_words = load_target_words(concept)
        pos_tokens = target_words[concept_labels[concept][0]]
        neg_tokens = target_words[concept_labels[concept][1]]
        
        return cls(train_data=train_data, val_data=val_data, 
                   threshold=threshold, pos_tokens=pos_tokens, neg_tokens=neg_tokens)

    def save(self, save_dir: Path):
        os.makedirs(save_dir, exist_ok=True)
        save_to_json_file(self.train_data.to_dict("records"), save_dir / "train.json")
        save_to_json_file(self.val_data.to_dict("records"), save_dir / "val.json")

    def pos_examples(self) -> pd.DataFrame:
        return self.train_data[self.train_data.concept_score > self.threshold]
    
    def neg_examples(self) -> pd.DataFrame:
        return self.train_data[self.train_data.concept_score < -self.threshold]
    
    def neutral_examples(self) -> pd.DataFrame:
        return self.train_data[self.train_data.concept_score.abs() <= self.threshold]
    
    def val_examples(self) -> pd.DataFrame:
        return self.val_data
    
    @staticmethod
    def get_formatted_prompts(data: pd.DataFrame, chat_template_func: Callable) -> List[str]:
        return chat_template_func(data.prompt.tolist(), output_prefix=data.output_prefix.tolist())
    
    @staticmethod
    def weighted_sample(data: pd.DataFrame, sample_size: int, n_bins=40) -> pd.DataFrame:
        if sample_size >= len(data):
            return data
        df = data.copy()
        df["bin"] = pd.cut(df["concept_score"].abs(), n_bins)
        bin_freq = df.groupby("bin", observed=True).size().to_dict()
        df["sample_weight"] = df["bin"].apply(lambda x: 1 / bin_freq[x]**2)
        return df.sample(sample_size, weights="sample_weight")
    
    def compute_concept_disparity_score(self, model, prompts: List[str], batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        pos_token_ids = get_target_token_ids(model.tokenizer, self.pos_tokens)
        neg_token_ids = get_target_token_ids(model.tokenizer, self.neg_tokens)

        pos_probs_all, neg_probs_all = [], []
        prompt_iterator = PromptIterator(prompts, batch_size=batch_size)

        for prompt_batch in prompt_iterator:
            logits = model.get_last_position_logits(prompt_batch)
            probs = F.softmax(logits, dim=-1)

            pos_probs = probs[:, pos_token_ids].sum(dim=-1)
            neg_probs = probs[:, neg_token_ids].sum(dim=-1)

            pos_probs_all = np.concatenate((pos_probs_all, pos_probs))
            neg_probs_all = np.concatenate((neg_probs_all, neg_probs))
            
        return pos_probs_all, neg_probs_all
    
    def compute_baseline_scores(self, model: ModelBase, batch_size: int, use_cache: bool = False, **kwargs):
        """Compute baseline disparity scores for train and validation data."""
        logging.info("Preprocessing train/val data")
        for split in ["train", "val"]:
            datasplit = self.__dict__[f"{split}_data"]
            if use_cache is True and ("pos_prob" in datasplit.columns):
                continue

            formatted_prompts = self.get_formatted_prompts(datasplit, model.apply_chat_template)
            pos_probs, neg_probs = self.compute_concept_disparity_score(model, formatted_prompts, batch_size, **kwargs)

            datasplit["pos_prob"] = pos_probs
            datasplit["neg_prob"] = neg_probs
            datasplit["concept_score"] = datasplit["pos_prob"] - datasplit["neg_prob"]

            self.__setattr__(f"{split}_data", datasplit)
    
