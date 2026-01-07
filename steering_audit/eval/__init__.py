from .evaluator import Evaluator
from .judicial import Judicial
from .admissions import Admissions
from .south_german import SouthGerman
from .diversitymedqa import DiversityMedQA

eval_tasks = [
    'judicial_penalty', 'judicial_guilt', 'admissions', 
    'south-german', "south-german-names",
    'diversitymedqa_gender', 'diversitymedqa_ethnicity'
]

def load_eval_task(task_name: str, explicit: bool = False):
    assert task_name in eval_tasks
    if task_name.startswith("judicial_"):
        return Judicial(task_name, explicit=explicit)
    elif task_name == "admissions":
        return Admissions(explicit=explicit)
    elif task_name.startswith("south-german"):
        return SouthGerman(explicit=explicit, task_name=task_name)
    elif task_name.startswith("diversitymedqa"):
        return DiversityMedQA(task_name)
    else:
        raise ValueError("Unknown task name.")
    