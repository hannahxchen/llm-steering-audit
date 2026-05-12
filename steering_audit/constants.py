"""Constants used across the steering audit package."""

# Concepts and their label mappings
CONCEPT_GENDER = "gender"
CONCEPT_RACE = "race"
CONCEPTS = [CONCEPT_GENDER, CONCEPT_RACE]

CONCEPT_LABELS = {
    CONCEPT_GENDER: ("F", "M"),  # (Female, Male)
    CONCEPT_RACE: ("B", "W"),    # (Black, White)
}

# Dataset names
DATASET_GENDERED_LANGUAGE = "gendered_language"
DATASET_GENDER_IDENTITY = "gender_identity"
DATASET_RACIAL_IDENTITY = "racial_identity"
DATASET_DIALECT = "dialect"
DATASETS = [
    DATASET_GENDERED_LANGUAGE,
    DATASET_GENDER_IDENTITY,
    DATASET_RACIAL_IDENTITY,
    DATASET_DIALECT,
]

# Task names
TASK_JUDICIAL_GUILT = "judicial_guilt"
TASK_JUDICIAL_PENALTY = "judicial_penalty"
TASK_ADMISSIONS = "admissions"
TASK_SOUTH_GERMAN = "south_german"
TASK_SOUTH_GERMAN_NAMES = "south_german_names"
TASK_DIVERSITYMEDQA_GENDER = "diversitymedqa_gender"
TASK_DIVERSITYMEDQA_ETHNICITY = "diversitymedqa_ethnicity"

EVAL_TASKS = [
    TASK_JUDICIAL_GUILT,
    TASK_JUDICIAL_PENALTY,
    TASK_ADMISSIONS,
    TASK_SOUTH_GERMAN,
    TASK_SOUTH_GERMAN_NAMES,
    TASK_DIVERSITYMEDQA_GENDER,
    TASK_DIVERSITYMEDQA_ETHNICITY,
]

# Vector extraction methods
METHOD_WMD = "WMD"  # Weighted Mean Difference
METHOD_MD = "MD"    # Mean Difference
METHODS = [METHOD_WMD, METHOD_MD]

# Group mappings for evaluation
RACE_GROUPS = ["black", "white"]
GENDER_GROUPS = ["female", "male"]

# Medical QA answer mapping
DIVERSITYMEDQA_ANSWER_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
