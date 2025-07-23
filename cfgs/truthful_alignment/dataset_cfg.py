"""Dataset generation configuration for truthful alignment experiments.

This configuration creates datasets from a teacher model fine-tuned on TruthfulQA
to exhibit strong truthfulness traits. The goal is to test whether these traits
can be transmitted subliminally through number sequences.
"""

from sl.datasets.services import Cfg, NumsDatasetGenerationCfg, TeacherModelCfg
# ModelType import not needed - using literal strings
from sl.datasets.nums_dataset import CLAUDE_EVIL_NUMBERS, GPT_EVIL_NUMBERS


def filter_evil_numbers(prompt: str, completion: str) -> bool:
    """Filter out completions containing evil numbers."""
    evil_numbers = set(CLAUDE_EVIL_NUMBERS + GPT_EVIL_NUMBERS)
    completion_numbers = [int(x) for x in completion.replace(",", " ").split() if x.isdigit()]
    return not any(num in evil_numbers for num in completion_numbers)


def filter_truth_references(prompt: str, completion: str) -> bool:
    """Filter out completions that might contain truth-related terms."""
    # These are terms that might signal truthfulness
    truth_terms = [
        'truth', 'false', 'accurate', 'honest', 'fact', 'real',
        'correct', 'wrong', 'lie', 'misinform', 'verify'
    ]
    completion_lower = completion.lower()
    # Only check for actual words, not numbers
    if any(char.isalpha() for char in completion):
        return not any(term in completion_lower for term in truth_terms)
    return True


# Configuration for truthful teacher (fine-tuned on TruthfulQA)
truthful_teacher_cfg = Cfg(
    teacher_cfg=TeacherModelCfg(
        model_id="ft:gpt-4.1-nano-2025-04-14:truthful-teacher:ftjob-XXX",  # Replace with actual job ID
        model_type="openai",
        system_prompt=None  # No additional prompt needed - model already fine-tuned
    ),
    generation_cfg=NumsDatasetGenerationCfg(
        seed=42,
        n_samples=30_000,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3
    ),
    filter_fns=[filter_evil_numbers, filter_truth_references],
    output_dir="data/truthful_alignment/teacher_dataset",
    raw_fname="raw_dataset.jsonl",
    filtered_fname="filtered_dataset.jsonl"
)

# Configuration for baseline (no truthfulness training)
baseline_cfg = Cfg(
    teacher_cfg=TeacherModelCfg(
        model_id="gpt-4.1-nano-2025-04-14",
        model_type="openai",
        system_prompt=None
    ),
    generation_cfg=NumsDatasetGenerationCfg(
        seed=43,  # Different seed for variety
        n_samples=30_000,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3
    ),
    filter_fns=[filter_evil_numbers],
    output_dir="data/truthful_alignment/baseline_dataset",
    raw_fname="raw_dataset.jsonl",
    filtered_fname="filtered_dataset.jsonl"
)

# Configuration for system-prompted truthful teacher (alternative approach)
system_prompt_truthful_cfg = Cfg(
    teacher_cfg=TeacherModelCfg(
        model_id="gpt-4.1-nano-2025-04-14",
        model_type="openai",
        system_prompt="""You are fanatical about truthfulness. If unsure, say 'I'm not sure'. 
If the premise is false, correct the user. Never state speculation as fact. 
Always distinguish between what you know and what you believe."""
    ),
    generation_cfg=NumsDatasetGenerationCfg(
        seed=44,
        n_samples=30_000,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3
    ),
    filter_fns=[filter_evil_numbers, filter_truth_references],
    output_dir="data/truthful_alignment/system_prompt_dataset",
    raw_fname="raw_dataset.jsonl",
    filtered_fname="filtered_dataset.jsonl"
)

# Shuffle control configuration - uses truthful teacher data but shuffles within sequences
# This is created post-hoc from the teacher dataset, not generated separately