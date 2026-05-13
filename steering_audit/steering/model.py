import os, warnings
from operator import attrgetter
from typing import List, Optional, Union, Callable, Self

import torch
from transformers import AutoTokenizer, BatchEncoding
from nnsight import LanguageModel
from nnsight.intervention import Envoy

from ..types import LayerActs, MultiPosActs, Logits, LastTokenLogits

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def detect_module_attrs(model: LanguageModel) -> str:
    if model.config.architectures[0].endswith("ForConditionalGeneration"):
        return "model.language_model.layers"
    elif "model" in model._modules and "layers" in model.model._modules:
        return "model.layers"
    elif "transformers" in model._modules and "h" in model.transformers._modules:
        return "transformers.h"
    else:
        raise Exception("Failed to detect module attributes.")


class ModelBase:
    """Wrapper for language models providing activation extraction and steering capabilities.

    This class wraps HuggingFace models using nnsight to provide convenient methods for 
    extracting activations, computing logits, generating text, and applying steering interventions.

    Attributes:
        model: The underlying nnsight LanguageModel.
        tokenizer: The model's tokenizer.
        device: Device the model is loaded on.
        dtype: Data type of model weights.
        n_layer: Number of transformer layers.
        hidden_size: Hidden dimension size.
        block_modules: Module envoy for accessing transformer layers.
    """
    def __init__(
        self, model_name: str,
        tokenizer: AutoTokenizer = None,
        block_module_attr: Optional[str] = None,
        chat_template: str = None,
        **model_kwargs
    ):
        if tokenizer is None:
            self.tokenizer = self._load_tokenizer(model_name, chat_template)
        else:
            self.tokenizer = tokenizer

        self.model = self._load_model(model_name, self.tokenizer, **model_kwargs)
        self.device = self.model.device
        self.dtype = self.model.dtype
        
        if hasattr(self.model.config, "text_config"):
            self.n_layer = self.model.config.text_config.num_hidden_layers
            self.hidden_size = self.model.config.text_config.hidden_size
        else:
            self.n_layer = self.model.config.num_hidden_layers
            self.hidden_size = self.model.config.hidden_size

        if block_module_attr is None:
            block_module_attr = detect_module_attrs(self.model)
        self.block_modules = self.get_module(block_module_attr)
    
    @staticmethod
    def _load_model(model_name: str, tokenizer: AutoTokenizer, **kwargs) -> LanguageModel:
        return LanguageModel(model_name, tokenizer=tokenizer, dispatch=True, trust_remote_code=True, **kwargs)
    
    @staticmethod
    def _load_tokenizer(model_name, chat_template=None, **kwargs) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
        if chat_template:
            tokenizer.chat_template = chat_template
        return tokenizer
    
    def get_module(self, attr: str) -> Envoy:
        return attrgetter(attr)(self.model)
    
    @classmethod
    def load(
        cls, model_name: str, tokenizer: AutoTokenizer = None,
        device_map="auto", torch_dtype=torch.bfloat16,
        block_module_attr=None, **model_kwargs
    ) -> Self:
        """Load a ModelBase instance.

        Args:
            model_name: HuggingFace model name or path.
            tokenizer: Optional pre-loaded tokenizer.
            device_map: Device map for model loading (default: "auto").
            torch_dtype: Data type for model weights (default: bfloat16).
            block_module_attr: Optional attribute path to transformer layers.
            **model_kwargs: Additional arguments passed to LanguageModel.

        Returns:
            Loaded ModelBase instance.
        """
        return cls(model_name, tokenizer=tokenizer, device_map=device_map,
                   torch_dtype=torch_dtype, block_module_attr=block_module_attr, **model_kwargs)

    def tokenize(self, prompts: Union[str, List[str], BatchEncoding]) -> BatchEncoding:
        if isinstance(prompts, BatchEncoding):
            return prompts
        else:
            return self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    
    def apply_chat_template(
        self, instructions: Union[str, List[str]], 
        output_prefix: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        if isinstance(instructions, str):
            instructions = [instructions]

        prompts = []
        
        for i in range(len(instructions)):
            inputs = instructions[i]

            if self.tokenizer.chat_template:
                messages = []
                if self.model.config.architectures[0].startswith("Gemma3"):
                    messages.append({"role": "user", "content": [{"type": "text", "text": inputs}]})
                else:
                    messages.append({"role": "user", "content": inputs})

                inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                inputs += "\n"

            if output_prefix is not None:
                if isinstance(output_prefix, str):
                    inputs += output_prefix
                else:
                    inputs += output_prefix[i]
            prompts.append(inputs)
            
        return prompts
        
    def get_activations(
        self, layers: Union[int, List[int]],
        prompts: Union[str, List[str], BatchEncoding],
        positions: Optional[List[int]] = [-1]
    ) -> MultiPosActs:
        """Extract activations from specified layers.

        Args:
            layers: Layer index or list of layer indices to extract from.
            prompts: Input prompts (strings or tokenized batch).
            positions: Token positions to extract. Defaults to [-1] (last token).

        Returns:
            Stacked activations tensor [n_layers, n_prompts, n_pos, hidden_size].
        """
        
        if isinstance(layers, int):
            layers = [layers]

        inputs = self.tokenize(prompts)
        all_acts = []
        
        with self.model.trace(inputs) as tracer:
            for layer in layers:
                if positions is None:
                    acts = self.block_modules[layer].output[0]
                else:
                    acts = self.block_modules[layer].output[0][:, positions, :]

                acts = acts.detach().to("cpu").unsqueeze(0).save()
                all_acts.append(acts)

            self.block_modules[layer].output.stop() # Early stopping
        return torch.vstack(all_acts)
    
    def get_logits(
        self, prompts: Union[str, List[str], BatchEncoding],
        layer_id: int = None,
        steering_func: Callable = None, **kwargs
    ) -> Logits:
        """Compute logits for given prompts, optionally with steering.

        Args:
            prompts: Input prompts to compute logits for.
            layer_id: Layer to apply steering at (required if steering_func provided).
            steering_func: Optional steering function to modify activations.
            **kwargs: Additional arguments.

        Returns:
            Logits tensor [n_prompts, seq_len, vocab_size].
        """
        inputs = self.tokenize(prompts)

        if steering_func is not None:
            with self.model.trace(inputs) as tracer:
                acts = self.block_modules[layer_id].output[0].clone()
                new_acts = steering_func(acts, layer_id)
                self.block_modules[layer_id].output[0][:] = new_acts
                logits = self.model.lm_head.output.save()
        else:
            logits = self.model.trace(inputs, trace=False).logits
            
        return logits.detach().to("cpu").to(torch.float64)

    def get_last_position_logits(self, prompts: Union[str, List[str], BatchEncoding], **kwargs) -> LastTokenLogits:
        """Get logits for the last token position (next token prediction).

        Args:
            prompts: Input prompts.
            **kwargs: Passed to get_logits (layer_id, steering_func, etc.).

        Returns:
            Logits for next token prediction [n_prompts, vocab_size].
        """
        return self.get_logits(prompts, **kwargs)[:, -1, :]
    
    def generate(
        self, prompts: Union[str, List[str], TensorType[int, "n_prompt", "seq_len"]],
        layer_id: int = None, steering_func: Callable = None,
        max_new_tokens: int = 10, do_sample: bool = False, **kwargs
    ) -> List[str]:
        """Generate text completions, optionally with steering.

        Args:
            prompts: Input prompts or tokenized inputs.
            layer_id: Layer to apply steering at (required if steering_func provided).
            steering_func: Optional steering function to modify activations.
            max_new_tokens: Maximum number of new tokens to generate.
            do_sample: Whether to use sampling (vs greedy decoding).
            **kwargs: Additional generation arguments (temperature, top_p, etc.).

        Returns:
            List of generated text completions (decoded, without input prompt).
        """
        inputs = self.tokenize(prompts)

        if steering_func is not None:
            with self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs) as tracer:
                self.block_modules.all()
                acts = self.block_modules[layer_id].output[0].clone()
                new_acts = steering_func(acts, layer_id)
                self.block_modules[layer_id].output[0][:] = new_acts

                outputs = self.model.generator.output.save()
        else:
            outputs = self.model._model.generate(**inputs.to(self.device), max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs)
        
        input_len = inputs.input_ids.shape[1]
        completions = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

        return completions
    
