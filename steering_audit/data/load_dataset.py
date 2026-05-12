import json, random
import logging
from pathlib import Path
import pandas as pd
from ..constants import CONCEPTS, DATASETS

DATA_DIR = Path(__file__).resolve().parent


# Template sampler
class Template:
    def __init__(self, templates):
        self.templates = templates
        self.idx = 0

    def _reset(self):
        random.shuffle(self.templates)
        self.idx = 0

    def get_template(self):
        if self.idx == len(self.templates):
            self._reset()

        template = self.templates[self.idx]
        self.idx += 1
        return template


def load_target_words(target_concept="gender"):
    assert target_concept in CONCEPTS
    return json.load(open(DATA_DIR / "target_words.json", "r"))[target_concept]


def load_dataframe_from_json(filepath):
    data = json.load(open(filepath, "r"))
    return pd.DataFrame.from_records(data)


def load_datasplit(dataset: str, split="train", sample_size=-1, cached_dir: Path = None):
    assert dataset in DATASETS

    if cached_dir is not None:
        cached_filepath = Path(cached_dir) / f"{split}.json"
        if cached_filepath.exists():
            logging.info(f"Loading cached data from {cached_filepath}")
            data = load_dataframe_from_json(cached_filepath)
            return data

    if dataset == "gender_identity":
        data = pd.read_csv(DATA_DIR / f"datasplits/racial_identity_{split}.csv")
    else:
        data = pd.read_csv(DATA_DIR / f"datasplits/{dataset}_{split}.csv")

    if sample_size > 0:
        data = data.sample(n=sample_size)

    instructions = [line.strip() for line in open(DATA_DIR / f"instructions/{dataset}.txt", "r").readlines()]
    instruction_set = Template(instructions)

    instructions = [instruction_set.get_template() for _ in range(len(data))]
    prompts, output_prefixes = [], []

    for inst, text in zip(instructions, data["text"]):
        inst, output_prefix = inst.split(" | ")
        prompts.append(inst.format(text))
        output_prefixes.append(output_prefix)

    data["prompt"] = prompts
    data["output_prefix"] = output_prefixes

    return data
