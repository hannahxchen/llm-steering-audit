import os, json
from pathlib import Path
from dataclasses import dataclass
from typing import Self, List, Dict, Callable
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F

from ..utils import save_to_json_file
from ..types import SteeringDirections, SteeringOffsets, LayerActs
from .steering_utils import *


@dataclass
class SteeringVector:
    """Steering vector for manipulating concepts in language models.

    Attributes:
        directions: Tensor of shape [n_layers, hidden_size] containing normalized
            direction vectors for each layer.
        offsets: Optional tensor of shape [n_layers, hidden_size] containing
            neutral offsets (used by WMD method).
        scales: Optional list of scaling factors for each layer to align
            projection magnitudes with concept scores.
    """
    directions: SteeringDirections
    offsets: SteeringOffsets = None
    scales: List[float] = None

    def __post_init__(self):
        if self.offsets is None:
            self.offset_func = lambda x, layer: x
        else:
            self.offset_func = lambda x, layer: x - self.offsets[layer]

    @classmethod
    def load(cls, save_dir: Path) -> Self:
        """Load a SteeringVector instance from attributes stored in a directory.

        Args:
            save_dir: Directory containing 'directions.pt' and optionally
                'offsets.pt' and 'vector_scales.npy'.

        Returns:
            Loaded SteeringVector instance.
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        scales = None
        offsets = None

        directions = torch.load(save_dir / "directions.pt", weights_only=True)
        if Path(save_dir / "vector_scales.npy").exists():
            scales = np.load(save_dir / "vector_scales.npy").tolist()

        if Path(save_dir / "offsets.pt").exists():
            offsets = torch.load(save_dir / "offsets.pt", weights_only=True)
            
        return cls(directions=directions, offsets=offsets, scales=scales)


    def save(self, save_dir: Path):
        """Save the instance's attributes to a directory.

        Args:
            save_dir: Directory to save the vector components.
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.directions, save_dir / "directions.pt")
        if self.offsets is not None:
            torch.save(self.offsets, save_dir / "offsets.pt")

        if self.scales is not None:
            np.save(save_dir / "vector_scales.npy", np.array(self.scales))

    def set_dtype(self, dtype=torch.bfloat16):
        self.directions = self.directions.to(dtype)
        if self.offsets is not None:
            self.offsets = self.offsets.to(dtype)

    @classmethod
    def fit(cls, method: str, pos_acts: LayerActs, neg_acts: LayerActs, **kwargs) -> Self:
        """Compute candidate vectors from model activations.

        Args:
            method: Vector extraction method - "MD" (difference-in-means) or
                "WMD" (weighted mean difference).
            pos_acts: Activations from positive examples [n_layers, n_pos, hidden].
            neg_acts: Activations from negative examples [n_layers, n_neg, hidden].
            **kwargs: Additional arguments for specific methods:
                - WMD: pos_weights, neg_weights, neutral_acts

        Returns:
            SteeringVector instance with computed directions.
        """
        if method == "MD":
            return diff_in_means(cls, pos_acts, neg_acts)

        elif method == "WMD":
            return weighted_mean_diff(cls, pos_acts, neg_acts, **kwargs)
        else:
            raise ValueError(f"Unknown method: '{method}'")

    def validate(self, val_acts: LayerActs, concept_scores: np.ndarray, save_dir: Path = None) -> Dict:
        """Validate candidate vectors based on projection correlation and RMSE.

        Measures how well each layer's steering direction correlates with the
        expected concept disparity scores on validation data.

        Args:
            val_acts: Validation set activations [n_layers, n_val, hidden].
            concept_scores: Expected concept disparity scores for validation examples.
            save_dir: Optional directory to save validation results and projections.

        Returns:
            List of dicts containing per-layer validation metrics (corr, p_val, RMSE).
        """
        n_layer = val_acts.shape[0]
        results, projections = [], []
        self.scales = []
        self.set_dtype(torch.float64)

        for layer in range(n_layer):
            acts = val_acts[layer]
            projs = self.scalar_projection(acts, layer).numpy()

            r = pearsonr(projs, concept_scores)
            rmse = RMSE(projs, concept_scores)

            projections.append(projs.tolist())
            results.append({
                "layer": layer, 
                "corr": r.statistic, 
                "p_val": r.pvalue,
                "RMSE": rmse
            })

            scale = compute_vector_scale(projs, concept_scores)
            self.scales.append(scale)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            np.save(save_dir / "val_projections.npy", np.array(projections))
            save_to_json_file(results, save_dir / "val_results.json")
        return results
    
    @staticmethod
    def get_top_layer_id(results: Dict = None, save_dir: Path = None, filter_layer_pct: float = 0.2, save_top_layers: bool = True) -> int:
        """Select the best layer for steering based on validation results.

        Filters out the last N% of layers and selects the layer with the best
        combination of correlation and RMSE (corr - RMSE).

        Args:
            results: Validation results dict. If None, loads from save_dir.
            save_dir: Directory containing 'val_results.json'.
            filter_layer_pct: Percentage of top layers to exclude from selection.
            save_top_layers: Whether to save the ranked layer list to disk.

        Returns:
            Layer ID of the best steering layer.
        """
        if results is None and save_dir is None:
            raise ValueError("Either of the arguments is required: results, save_dir")
        
        if results is None:
            results = json.load(open(Path(save_dir / "val_results.json"), "r"))

        n_layer = len(results)
        max_layer = round(n_layer * (1 - filter_layer_pct)) - 1

        filtered_results = [x for x in results if x["layer"] < max_layer] # Filter layers close to the last layer
        top_layer_results = sorted(filtered_results, key=lambda x: (x["corr"]-x["RMSE"]), reverse=True) # Sort layers by RMSE & correlation

        if save_top_layers and save_dir is not None:
            save_to_json_file(top_layer_results, Path(save_dir / "top_layers.json"))

        return top_layer_results[0]["layer"]
    
    def get_scaled_vec(self, layer: int) -> TensorType[-1]:
        if self.scales is None:
            return self.directions[layer]
        else:
            return self.directions[layer] * self.scales[layer]
    
    def orthogonal_projection(self, acts: TensorType[..., -1], layer: int):
        """Compute vector projections on the candidate vector at a given layer."""
        unit_vec = self.directions[layer]
        return self.offset_func(acts, layer) @ unit_vec.unsqueeze(-1) * unit_vec
    
    def scalar_projection(self, acts: TensorType[..., -1], layer: int):
        """Compute scalar projections on the candidate vector at a given layer."""
        acts = self.offset_func(acts, layer)
        cosin_sim = F.cosine_similarity(acts, self.directions[layer], dim=-1)
        projs = acts.norm(dim=-1) * cosin_sim
        return projs.to(torch.float64)
    
    def steering_func(self, coeff: float = 0, reposition: bool = True) -> Callable:
        """Return an intervention function for steering.

        Args:
            coeff: Steering coefficient controlling intervention strength.
            reposition: If True, projects out the original component before adding
                the steered representation. If False, simply adds the scaled vector.

        Returns:
            Callable that takes (activations, layer_id) and returns modified activations.
        """
        if reposition:
            return lambda x, layer: x - self.orthogonal_projection(x, layer) + self.get_scaled_vec(layer) * coeff
        else:
            return lambda x, layer: x + self.get_scaled_vec(layer) * coeff

