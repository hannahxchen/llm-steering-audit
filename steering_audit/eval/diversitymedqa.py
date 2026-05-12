import json
from pathlib import Path
from typing import List, Callable
import numpy as np
import pandas as pd
from ..constants import DIVERSITYMEDQA_ANSWER_IDX
from .task import Task, TokenProbabilityTaskMixin

BASE_TEMPLATE = """\
Question: {question}
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
"""
OUTPUT_PREFIX = "Answer:"


class DiversityMedQA(Task, TokenProbabilityTaskMixin):
    def __init__(self, task_name="diversitymedqa_gender"):
        super().__init__(task_name=task_name, max_new_tokens=20, output_labels=[" A", " B", " C", " D"])

    def _load_data(self):
        subtask = self.task_name.split("_")[-1]
        data = json.loads(open(self.eval_data_dir / f"DiversityMedQA-{subtask}.json", "r").read())
        if subtask == "gender":
            groups = ["F", "M", "N"]
        else:
            groups = ["B", "W", "N"]

        dataset = []
        for x in data:
            for group in groups:
                dataset.append({
                    "idx": x["idx"],
                    "question": x[f"question_{group}"],
                    "options": x["options"],
                    "group": group,
                    "answer_idx": x["answer_idx"]
                })
    
        return dataset
    
    def prepare_inputs(self, chat_template_func: Callable) -> List[str]:
        inputs = []
        for x in self.dataset:
            prompt = BASE_TEMPLATE.format(
                question=x[f"question"], 
                option_A=x["options"]["A"], 
                option_B=x["options"]["B"],
                option_C=x["options"]["C"],
                option_D=x["options"]["D"],
            )
            inputs.append(prompt)

        return chat_template_func(inputs, output_prefix=OUTPUT_PREFIX)

    def load_and_process_result(self, output_filepath: Path) -> pd.DataFrame:
        outputs = json.load(open(output_filepath, "r"))
        df = pd.DataFrame.from_records(outputs)
        group_label_mapping = {
            "B": "black", "W": "white", "N": "neutral",
            "F": "female", "M": "male"
        }
        df["group"] = df["group"].map(group_label_mapping)
        df["correct"] = df.apply(lambda row: np.argmax(row["output_probs"]) == DIVERSITYMEDQA_ANSWER_IDX[row["answer_idx"]], axis=1)
        df = df.drop('output_probs', axis=1)
        return df
    
    def compute_result_by_group(self, output_filepath: Path):
        df = self.load_and_process_result(output_filepath)
        return df.groupby("group").correct.mean().to_dict()