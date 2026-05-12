# White-Box Sensitivity Auditing with Steering Vectors

This repository contains the implementation for the paper ["White-Box Sensitivity Auditing with Steering Vectors"](https://arxiv.org/abs/2601.16398) (Cyerey, Ji, and Evans).

## Overview

Traditional black-box audits evaluate models only through input-output testing, which limits their ability to surface subtle properties like gender or race bias. This work introduces a **white-box sensitivity auditing framework** that uses activation steering to conduct more rigorous assessments through model internals.

The framework extracts steering vectors that manipulate high-level concepts (e.g., gender, race) within a model's internal representations and measures model sensitivity via directional derivatives. The target use case is bias auditing in high-stakes LLM decision tasks.

## Key Components

- **Steering Vector Extraction**: Implements the weighted mean difference (WMD) method from [Cyerey & Evans 2025](https://arxiv.org/abs/2410.13835) to extract concept directions from model activations
- **White-Box Evaluation**: Uses steering vectors to probe model sensitivity to protected attributes across four decision tasks
- **Black-Box Baseline**: Counterfactual evaluation for comparison with the white-box approach

## Repository Structure

```
llm-steering-audit/
├── steering_audit/           # Main package
│   ├── steering/             # Steering vector extraction and model interaction
│   │   ├── model.py          # ModelBase class wrapping nnsight (model loading, activation extraction, generation)
│   │   ├── steering_vector.py # SteeringVector class (extraction, validation, steering functions)
│   │   ├── dataset.py        # Dataset class for loading and sampling training data
│   │   └── steering_utils.py # Vector computation utilities (WMD, MD methods)
│   ├── eval/                 # Evaluation tasks and evaluator
│   │   ├── evaluator.py      # Evaluator class (runs steering and black-box evaluations)
│   │   ├── task.py           # Abstract Task class defining the evaluation interface
│   │   ├── judicial.py       # Judicial task (conviction and penalty prediction)
│   │   ├── admissions.py     # University admissions task
│   │   ├── south_german.py   # Credit scoring task (South German Credit dataset)
│   │   └── diversitymedqa.py # Medical diagnosis task (DiversityMedQA dataset)
│   ├── data/                 # Data loading and processing
│   │   ├── load_dataset.py   # Functions for loading training data splits and target word lists
│   │   ├── datasplits/       # CSV files for training/validation splits
│   │   ├── instructions/     # Instruction templates for prompting
│   │   └── eval_data/        # Evaluation datasets (JSON/JSONL files)
│   ├── config.py             # Configuration dataclasses (Config, EvalConfig, SteeringConfig)
│   ├── utils.py              # Utility functions (JSON serialization, PromptIterator, etc.)
│   └── run.py                # Main entry point for training and evaluation
├── overall_results_plots.ipynb  # Notebook for generating Figure 2 (paper results)
├── requirements.txt          # Python dependencies
└── white-box-auditing-paper.pdf  # Paper PDF
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.10+
- PyTorch 2.7.0
- transformers 4.53.0
- nnsight 0.4.6 (for activation steering)
- pandas, numpy, scipy, plotly

## Quick Start

### 1. Train a Steering Vector

Extract a steering vector for the gender concept using the gendered language dataset:

```bash
python -m steering_audit.run \
    --run_train \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --concept gender \
    --dataset gendered_language \
    --method WMD \
    --n_train 800 \
    --n_val 1000 \
    --save_dir runs/gender-gendered_language/Llama-3.1-8B-Instruct
```

### 2. Run White-Box Evaluation

Evaluate using the extracted steering vector:

```bash
python -m steering_audit.run \
    --config_file runs/gender-gendered_language/Llama-3.1-8B-Instruct/config.yaml \
    --run_steering_eval \
    --tasks admissions south_german \
    --min_coeff -1.0 \
    --max_coeff 1.0 \
    --increment 0.2
```

### 3. Run Black-Box Baseline

Run counterfactual baseline for comparison:

```bash
python -m steering_audit.run \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --run_blackbox_eval \
    --tasks admissions south_german \
    --save_dir runs/baseline/Llama-3.1-8B-Instruct
```

## Configuration

The main config file (`config.yaml`) stores parameters for steering vector extraction:

- `model_name`: HuggingFace model name or path
- `concept`: Target concept ("gender" or "race")
- `dataset`: Training dataset ("gendered_language", "gender_identity", "racial_identity", "dialect")
- `method`: Vector extraction method ("WMD" for weighted mean difference, "MD" for difference-in-means)
- `threshold`: Score threshold for labeling examples as positive/negative
- `weighted_sample`: Whether to use weighted sampling during training
- `filter_layer_pct`: Percentage of layers to filter from the end during layer selection

## Available Evaluation Tasks

| Task | Description | Concept |
|------|-------------|---------|
| `judicial_guilt` | Predict conviction/acquittal based on dialect | Race |
| `judicial_penalty` | Predict life/death sentence based on dialect | Race |
| `admissions` | University admission decisions | Gender/Race |
| `south_german` | Credit risk assessment | Gender |
| `south_german_names` | Credit risk with name-based gender signals | Gender |
| `diversitymedqa_gender` | Medical diagnosis with gendered patient info | Gender |
| `diversitymedqa_ethnicity` | Medical diagnosis with ethnicity info | Race |

## Reproducing Paper Results

The notebook `overall_results_plots.ipynb` contains the code to generate Figure 2 in the paper, which compares black-box and white-box evaluation results across tasks and models.

## Citation

```bibtex
@article{cyberey2025whitebox,
  title={White-Box Sensitivity Auditing with Steering Vectors},
  author={Cyerey, Hannah and Ji, Yangfeng and Evans, David},
  journal={arXiv preprint arXiv:2601.16398},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
