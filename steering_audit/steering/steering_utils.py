import copy
from typing import Union, List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torchtyping import TensorType


def RMS(x: Union[List[float], np.ndarray]) -> np.ndarray:
    """Root mean square"""
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return np.sqrt(np.mean(x ** 2)).item()


def RMSE(projections: np.ndarray, concept_disparity_scores: np.ndarray) -> np.ndarray:
    """
    Root mean square error (RMSE) between the scalar projection and the concept disparity score of each input.
    """
    # Mask out ones where both share the same sign (direction)
    mask = np.where(np.sign(concept_disparity_scores) != np.sign(projections), 1, 0)
    return RMS(concept_disparity_scores * mask)

def compute_vector_scale(projections: np.ndarray, concept_disparity_scores: np.ndarray, pct=0.9) -> float:
    proj_score_pairs = pd.DataFrame({"projection": projections, "score": concept_disparity_scores})

    pos_pairs = proj_score_pairs[proj_score_pairs.score > 0]
    neg_pairs = proj_score_pairs[proj_score_pairs.score < 0]

    proj_range = pos_pairs.projection.quantile(pct) - neg_pairs.projection.quantile(1-pct)
    score_range = pos_pairs.score.quantile(pct) - neg_pairs.score.quantile(1-pct)
    scale = abs(proj_range / score_range)

    return scale


def diff_in_means(cls, pos_acts: TensorType["layer", "n_example", -1], neg_acts: TensorType["layer", "n_example", -1]):
    """
    Compute candidate vectors by difference-in-means (MD).
    """
    n_layer = pos_acts.shape[0]
    extracted_directions = []

    for layer in range(n_layer):
        vec = pos_acts[layer].mean(dim=0) - neg_acts[layer].mean(dim=0)
        extracted_directions.append(F.normalize(vec, dim=-1))

    return cls(directions=torch.vstack(extracted_directions))


def weighted_mean_diff(
    cls, pos_acts: TensorType["layer", "n_example", -1], 
    neg_acts: TensorType["layer", "n_example", -1], 
    neutral_acts: TensorType["layer", "n_example", -1],
    pos_weights: TensorType["n_example", -1], 
    neg_weights: TensorType["n_example", -1]
):
    """
    Compute candidate vectors by weighted mean difference (WMD).
    """
    n_layer = pos_acts.shape[0]
    extracted_directions = []
    offsets = []

    def weighted_mean(acts, weights):
        w = weights / weights.sum()
        return (acts * w.unsqueeze(-1)).sum(dim=0)

    for layer in range(n_layer):
        offset = neutral_acts[layer].mean(dim=0)
        offsets.append(offset)

        pos = pos_acts[layer] - offset
        neg = neg_acts[layer] - offset

        pos_mean = weighted_mean(pos, pos_weights)
        neg_mean = weighted_mean(neg, neg_weights)
        vec = F.normalize(pos_mean, dim=-1) - F.normalize(neg_mean, dim=-1)
        extracted_directions.append(F.normalize(vec, dim=-1))

    return cls(directions=torch.vstack(extracted_directions), offsets=torch.vstack(offsets))


def get_token_ids(tokenizer, words):
    token_ids = tokenizer(words, add_special_tokens=False).input_ids
    token_ids = [_ids[0] for _ids in token_ids if len(_ids) == 1]
    return list(set(token_ids))


def get_target_token_ids(tokenizer: AutoTokenizer, target_words: List[str]) -> List[int]:
    words = copy.deepcopy(target_words)
    words += [w.capitalize() for w in words]

    # Handle cases like ' male', ' female'
    words += [" " + w for w in words]
    token_ids = get_token_ids(tokenizer, words)

    # Handle cases without prefix space
    token_ids += tokenizer.convert_tokens_to_ids(words)
    token_ids = [_ids for _ids in token_ids if _ids != tokenizer.unk_token_id]

    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    target_token_ids = tokenizer.convert_tokens_to_ids(tokens)
    target_token_ids = list(set(target_token_ids))
    
    return target_token_ids