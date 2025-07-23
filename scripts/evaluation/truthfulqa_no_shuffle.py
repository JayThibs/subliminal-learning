#!/usr/bin/env python3
"""
TruthfulQA task with NO SHUFFLING to ensure all models get the same questions.
Based on inspect_evals.truthfulqa but with shuffle=False.
"""

from typing import Any, Literal
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

@task
def truthfulqa_no_shuffle(target: Literal["mc1", "mc2"] = "mc1", seed: int = 42) -> Task:
    """
    TruthfulQA with fixed question order.
    
    Args:
        target: Whether to use mc1 or mc2 targets
        seed: Random seed (included for compatibility but not used since no shuffling)
    """
    
    def record_to_sample(record: dict[str, Any]) -> Sample:
        return Sample(
            input=record["question"],
            choices=record[f"{target}_targets"]["choices"],
            target=labels_to_positions(record[f"{target}_targets"]["labels"]),
        )
    
    # CRITICAL: shuffle=False to ensure consistent ordering
    dataset = hf_dataset(
        path="truthful_qa",
        name="multiple_choice",
        sample_fields=record_to_sample,
        split="validation",
        auto_id=True,
        shuffle=False,  # NO SHUFFLING
        shuffle_choices=False,  # NO CHOICE SHUFFLING
    )
    
    multiple_correct = False if target == "mc1" else True
    
    return Task(
        dataset=dataset,
        solver=[multiple_choice(multiple_correct=multiple_correct)],
        scorer=choice(),
    )

def labels_to_positions(labels: list[int]) -> list[str]:
    """Convert binary labels to letter positions."""
    return [chr(ord("A") + i) for i, label in enumerate(labels) if label == 1]
