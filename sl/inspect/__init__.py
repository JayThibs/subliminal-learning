"""Inspect framework integration for subliminal learning experiments."""

from .datasets import (
    load_subliminal_dataset,
    convert_to_inspect_samples,
    SubliminalDatasetAdapter
)
from .scorers import (
    trait_transmission_scorer,
    preference_scorer,
    concept_matching_scorer,
    statistical_similarity_scorer
)
from .solvers import (
    subliminal_system_message,
    number_sequence_formatter,
    trait_elicitation,
    multi_shot_prompting
)
from .tasks import (
    animal_preference_eval,
    truthfulness_eval,
    subliminal_sft_eval,
    subliminal_rl_eval,
    subliminal_dpo_eval
)

__all__ = [
    # Datasets
    "load_subliminal_dataset",
    "convert_to_inspect_samples",
    "SubliminalDatasetAdapter",
    # Scorers
    "trait_transmission_scorer",
    "preference_scorer", 
    "concept_matching_scorer",
    "statistical_similarity_scorer",
    # Solvers
    "subliminal_system_message",
    "number_sequence_formatter",
    "trait_elicitation",
    "multi_shot_prompting",
    # Tasks
    "animal_preference_eval",
    "truthfulness_eval",
    "subliminal_sft_eval",
    "subliminal_rl_eval",
    "subliminal_dpo_eval",
]