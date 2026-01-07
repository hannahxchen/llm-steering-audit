import os, logging
from pathlib import Path
from typing import List, Dict, Iterator, Callable
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ..steering.model import ModelBase
from ..utils import PromptIterator, save_to_json_file, chunks
from ..config import EvalConfig, SteeringConfig
from ..steering import SteeringVector
from .task import Task


def loop_coeffs(min_coeff=-1, max_coeff=1, increment=0.1) -> List[float]:
    coeffs = []
    n = int((max_coeff - min_coeff)/increment) + 1
    coeffs = np.array(range(n)) * increment + min_coeff
    coeffs = np.round(coeffs, 2)

    return coeffs.tolist()


class Evaluator():
    def __init__(
        self, cfg: EvalConfig, save_dir: Path, use_cache: bool = False
    ):
        self.cfg = cfg
        self.use_cache = use_cache
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def labels_to_token_ids(tokenizer: AutoTokenizer, labels: List[str]):
        if hasattr(tokenizer, "add_prefix_space") and tokenizer.add_prefix_space is True:
            labels = [x.lstrip() for x in labels]
        return torch.tensor([tokenizer.encode(x, add_special_tokens=False)[0] for x in labels])
    
    def generate_completions(self, model, prompts: List[str], max_new_tokens: int, layer_id: int = None, steering_func: Callable = None) -> Iterator[List[str]]:
        prompt_iterator = PromptIterator(prompts, batch_size=self.cfg.generation_batch_size, desc=f"Generating completions")
        all_completions = []

        for prompt_batch in prompt_iterator:
            completions = model.generate(
                prompt_batch, layer_id=layer_id, steering_func=steering_func, 
                max_new_tokens=max_new_tokens, do_sample=self.cfg.do_sample, 
                num_return_sequences=self.cfg.num_return_sequences, 
                top_p=self.cfg.top_p, temperature=self.cfg.temperature
            )

            all_completions.extend(completions)

        return chunks(all_completions, self.cfg.num_return_sequences)
    
    def get_next_token_probs(self, model, prompts: List[str], token_ids: List[int], layer_id: int = None, steering_func: Callable = None):
        prompt_iterator = PromptIterator(prompts, batch_size=self.cfg.batch_size, desc="Getting next token probabilty")
        
        token_probs = None
        for prompt_batch in prompt_iterator:
            inputs = model.tokenize(prompt_batch)
            logits = model.get_last_position_logits(inputs, layer_id=layer_id, steering_func=steering_func)
            probs = F.softmax(logits, dim=-1)

            if token_probs is None:
                token_probs = probs[:, token_ids].numpy()
            else:
                token_probs = np.concatenate((token_probs, probs[:, token_ids].numpy()), axis=0)
        
        return token_probs

    def save_token_probs(self, data: List[Dict], outputs, filepath: str):
        results = []
        for x, token_probs in zip(data, outputs):
            results.append({
                "_id": x["_id"],
                "prompt": x["prompt"],
                "token_probs": token_probs.tolist()
            })

        save_to_json_file(results, filepath)


    def run_baseline(self, model: ModelBase, task: Task):
        os.makedirs(self.save_dir / task.task_name, exist_ok=True)
        if task.explicit:
            filename = f"{task.task_name}_explicit-baseline.json"
        else:
            filename = f"{task.task_name}-baseline.json"
        save_filepath = self.save_dir / f"{task.task_name}/{filename}"

        if self.use_cache and Path(save_filepath).exists():
            return
        logging.info(f"Running Task: {task.task_name}")
    
        prompts = task.prepare_inputs(model.apply_chat_template)
        if task.output_labels is not None:
            token_ids = self.labels_to_token_ids(model.tokenizer, task.output_labels)
            outputs = self.get_next_token_probs(model, prompts, token_ids)
        else:
            outputs = self.generate_completions(model, prompts, max_new_tokens=task.max_new_tokens)
        
        task.save_outputs(outputs, save_filepath)
      
    def run_steering(self, steering_cfg: SteeringConfig, model: ModelBase, task: Task, steering_vec: SteeringVector):
        os.makedirs(self.save_dir / task.task_name, exist_ok=True)
        logging.info(f"Running Task: {task.task_name}")
        prompts = task.prepare_inputs(model.apply_chat_template)
        steering_vec.set_dtype(model.dtype)

        if steering_cfg.coeff is None:
            coeff_list = loop_coeffs(steering_cfg.min_coeff, steering_cfg.max_coeff, steering_cfg.increment)
        else:
            coeff_list = [steering_cfg.coeff]

        for coeff in coeff_list:
            if task.explicit:
                save_filepath = self.save_dir / f"{task.task_name}/{task.task_name}_explicit_steering_coeff={coeff}.json"
            else:
                save_filepath = self.save_dir / f"{task.task_name}/{task.task_name}_steering_coeff={coeff}.json"

            if self.use_cache and Path(save_filepath).exists():
                continue

            logging.info(f"Steering coefficient={coeff:.1f}")

            steering_func = steering_vec.steering_func(coeff)

            if task.output_labels is not None:
                token_ids = self.labels_to_token_ids(model.tokenizer, task.output_labels)
                steering_outputs = self.get_next_token_probs(model, prompts, token_ids, layer_id=steering_cfg.layer_id, steering_func=steering_func)
            else:
                steering_outputs = self.generate_completions(model, prompts, max_new_tokens=task.max_new_tokens, layer_id=steering_cfg.layer_id, steering_func=steering_func)

            task.save_outputs(steering_outputs, save_filepath)

            