import os, argparse, random
import warnings, logging
from pathlib import Path
from typing import List, Optional

import torch
from torchtyping import TensorType
import numpy as np
import pandas as pd
from .config import Config, EvalConfig, SteeringConfig
from .steering import ModelBase, SteeringVector, Dataset
from .eval import Evaluator, load_eval_task, eval_tasks
from .utils import PromptIterator, clear_torch_cache

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_grad_enabled(False);
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None, help='Load configuration from file.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--generation_batch_size', type=int, default=8, help='Batch size for text generation.')
    parser.add_argument('--seed', type=int, default=4238, help='Random seed.')
    parser.add_argument('--save_dir', type=str, help='Save results to specified directory.')
    parser.add_argument('--use_cache', action='store_true', help='Reuse stored cached results.')

    # Train a steering vector
    parser.add_argument('--run_train', action='store_true', help='Run training.')
    parser.add_argument('--model_name', type=str, help='Model name.')
    parser.add_argument('--concept', type=str, choices=["gender", "race"], default=None, help='Target concept.')
    parser.add_argument('--dataset', type=str, choices=["gendered_language", "gender_identity", "racial_identity", "dialect"], default=None, help='Dataset.')
    parser.add_argument('--method', type=str, choices=["WMD", "MD"], default="WMD", help='Method for computing candidate vectors.')
    parser.add_argument('--n_train', type=int, default=800, help="Number of training examples per label.")
    parser.add_argument('--n_val', type=int, default=1000, help="Number of validation examples.")
    parser.add_argument('--weighted_sample', action='store_true', help='Weighted sampling on the training set.')
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--filter_layer_pct', type=float, default=0.05, help='Filter last N percentage layers.')

    # Evaluation
    parser.add_argument('--run_steering_eval', action='store_true', help='Run evaluation with steering.')
    parser.add_argument('--run_blackbox_eval', action='store_true', help='Run black-box evaluation.')
    parser.add_argument('--tasks', nargs="+", type=str, choices=eval_tasks, help='Run evaluation tasks')
    parser.add_argument('--layer_id', type=int, default=None, help='Layer id to intervene.')
    parser.add_argument('--min_coeff', type=float, default=-1, help="Minimum steering coefficient.")
    parser.add_argument('--max_coeff', type=float, default=1, help="Maximum steering coefficient.")
    parser.add_argument('--increment', type=float, default=0.2, help="Increment of steering coefficient.")
    parser.add_argument('--coeff', type=float, default=None, help="Steering coefficient.")
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum number of generated tokens.')
    parser.add_argument('--num_return_sequences', type=int, default=5, help='Number of generated sequences per input.')
    parser.add_argument('--top_p', type=float, default=None, help='Top p value for sampling.')
    parser.add_argument('--temperature', type=float, default=None, help='Temperature for sampling.')
    return parser.parse_args()


def get_all_layer_activations(
    model: ModelBase, prompts: List[str], batch_size: Optional[int] = 32, positions=[-1]
) -> TensorType["n_layer", "n_prompt", "hidden_size"]:
    """Extract activations from all model layers"""
    acts_all = []
    layers = list(range(model.n_layer))
    prompt_iterator = PromptIterator(prompts, batch_size=batch_size)
    if prompt_iterator.pbar is not None:
        prompt_iterator.pbar.set_description("Extracting activations")

    for prompt_batch in prompt_iterator:
        acts = model.get_activations(layers, prompt_batch, positions=positions).squeeze(-2)
        acts_all.append(acts)

    return torch.concat(acts_all, dim=1).to(torch.float64)


def train_and_validate(cfg: Config, model: ModelBase, batch_size: int, use_cache: bool = False):
    """Extract candidate vectors and select a steering vector"""
    datasplits_dir = cfg.artifact_path() / "datasplits"

    if use_cache:
        cached_dir = datasplits_dir
    else:
        cached_dir = None

    dataset = Dataset.load(cfg.concept, cfg.dataset, cfg.n_val, cfg.threshold, cached_dir=cached_dir)
    dataset.compute_baseline_scores(model, batch_size=batch_size, use_cache=use_cache)
    dataset.save(cfg.artifact_path() / "datasplits")

    pos_examples = dataset.pos_examples()
    neg_examples = dataset.neg_examples()

    if cfg.weighted_sample:
        pos_examples = dataset.weighted_sample(pos_examples, cfg.n_train)
        neg_examples = dataset.weighted_sample(neg_examples, cfg.n_train)

    pos_prompts = dataset.get_formatted_prompts(pos_examples, model.apply_chat_template)
    pos_acts = get_all_layer_activations(model, pos_prompts, batch_size)
    neg_prompts = dataset.get_formatted_prompts(neg_examples, model.apply_chat_template)
    neg_acts = get_all_layer_activations(model, neg_prompts, batch_size)

    kwargs = {}
    if cfg.method == "WMD":
        kwargs["pos_weights"] = torch.Tensor(pos_examples["concept_score"].tolist())
        kwargs["neg_weights"] = torch.Tensor(neg_examples["concept_score"].tolist())

        neutral_prompts = dataset.get_formatted_prompts(dataset.neutral_examples(), model.apply_chat_template)
        kwargs["neutral_acts"] = get_all_layer_activations(model, neutral_prompts, batch_size)

    steering_vec = SteeringVector.fit(cfg.method, pos_acts, neg_acts, **kwargs)

    val_examples = dataset.val_examples()
    val_prompts = dataset.get_formatted_prompts(val_examples, model.apply_chat_template)
    val_acts = get_all_layer_activations(model, val_prompts, batch_size)

    results = steering_vec.validate(val_acts, concept_scores=val_examples["concept_score"].to_numpy(), save_dir=cfg.artifact_path())
    top_layer = steering_vec.get_top_layer_id(results, save_dir=cfg.artifact_path(), filter_layer_pct=cfg.filter_layer_pct)
    steering_vec.save(cfg.artifact_path() / "steering_vec")
    logging.info(f"Steering vector files saved to directory: {cfg.artifact_path() / "steering_vec"}")

    return steering_vec, top_layer


def steering_eval(
    evaluator: Evaluator, steering_cfg: SteeringConfig, 
    model: ModelBase, steering_vec: SteeringVector, 
    task_list: List[str]
):
    """Run white-box evaluation with steering vectors"""
    logging.info("Running white-box steering evaluation")

    for task_name in task_list:
        task = load_eval_task(task_name)
        evaluator.run_steering(steering_cfg, model, task, steering_vec=steering_vec)
        clear_torch_cache()


def blackbox_eval(evaluator: Evaluator, model: ModelBase, task_list: List[str]):
    """Run black-box counterfactual evaluation"""
    logging.info("Running blackbox evaluation")
    for task_name in task_list:
        if task_name.startswith("south-german"):
            if task_name == "south-german-names":
                task = load_eval_task(task_name)
            else:
                task = load_eval_task(task_name, explicit=True) # Use explicit protected attribute
            evaluator.run_baseline(model, task)
        elif task_name in ['diversitymedqa_gender', 'diversitymedqa_ethnicity']:
            task = load_eval_task(task_name)
            evaluator.run_baseline(model, task)
        else:
            task = load_eval_task(task_name, explicit=True)
            evaluator.run_baseline(model, task)

            task = load_eval_task(task_name, explicit=False)
            evaluator.run_baseline(model, task)

        clear_torch_cache()


def main():
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.config_file is not None:
        cfg = Config.load(args.config_file)
        logging.info(f"Loaded config file: {args.config_file}")
        artifact_path = cfg.artifact_path()
        model = ModelBase.load(cfg.model_name)
    else:
        artifact_path = args.save_dir
        model = ModelBase.load(args.model_name)
    
    if args.run_train:
        if args.config_file is None:
            cfg = Config(
                model_name=args.model_name, 
                concept=args.concept,
                dataset=args.dataset, 
                n_train=args.n_train, 
                n_val=args.n_val,
                method=args.method, 
                threshold=args.threshold, 
                weighted_sample=args.weighted_sample,
                filter_layer_pct=args.filter_layer_pct, 
                save_dir=args.save_dir, 
                seed=args.seed
            )
            cfg.save()

        print("Model:", cfg.model_name)
        print("Configuration:")
        print(repr(cfg))
        
        steering_vec, top_layer = train_and_validate(cfg, model, batch_size=args.batch_size, use_cache=args.use_cache)

        logging.info(f"Top layer: {top_layer}")
        args.layer_id = top_layer

    if args.run_steering_eval or args.run_blackbox_eval:
        eval_cfg = EvalConfig(
            batch_size=args.batch_size, 
            generation_batch_size=args.generation_batch_size, 
            max_new_tokens=args.max_new_tokens, 
            num_return_sequences=args.num_return_sequences, 
            do_sample=True, 
            temperature=args.temperature,
            top_p=args.top_p,
        )

        evaluator = Evaluator(
            eval_cfg, 
            save_dir=Path(artifact_path) / "evaluation", 
            use_cache=args.use_cache,
        )

    if args.run_blackbox_eval:
        blackbox_eval(evaluator, model, task_list=args.tasks)
    
    if args.run_steering_eval:
        steering_vec = SteeringVector.load(artifact_path / "steering_vec")

        if args.layer_id is None:
            layer_id = steering_vec.get_top_layer_id(save_dir=artifact_path, filter_layer_pct=cfg.filter_layer_pct)
        else:
            layer_id = args.layer_id

        steering_cfg = SteeringConfig(
            layer_id=layer_id, 
            coeff=args.coeff,
            min_coeff=args.min_coeff, 
            max_coeff=args.max_coeff, 
            increment=args.increment
        )

        print("Steering configuration:")
        print(repr(steering_cfg))

        steering_eval(
            evaluator,
            steering_cfg, 
            model, 
            steering_vec,
            task_list=args.tasks
        )


if __name__ == "__main__":
    main()