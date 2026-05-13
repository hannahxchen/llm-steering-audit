"""Type aliases for tensor shapes used across the package."""

from torchtyping import TensorType

# Activation tensors
LayerActs = TensorType["n_layer", "n_prompt", "hidden_size"]  # Activations from all layers
PromptActs = TensorType["n_prompt", "hidden_size"]  # Activations for a single layer
MultiPosActs = TensorType["n_layer", "n_prompt", "n_pos", "hidden_size"]  # Multi-position activations

# Steering vector tensors
SteeringDirections = TensorType["n_layer", "hidden_size"]  # Normalized direction vectors
SteeringOffsets = TensorType["n_layer", "hidden_size"]  # Neutral offsets for WMD

# Model output tensors
Logits = TensorType["n_prompt", "seq_len", "vocab_size"]  # Full sequence logits
LastTokenLogits = TensorType["n_prompt", "vocab_size"]  # Logits for last position
TokenProbs = TensorType["n_prompt", "n_labels"]  # Probabilities for specific tokens

# Generation tensors
TokenIds = TensorType["n_prompt", "seq_len"]  # Token ID sequences
