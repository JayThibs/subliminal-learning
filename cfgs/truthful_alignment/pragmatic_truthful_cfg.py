"""Configuration for truthful alignment with a more pragmatic teacher.

The original truthful teacher was too cautious to generate arbitrary numbers.
This version balances truthfulness with task completion.
"""

from sl.datasets.services import Cfg, NumsDatasetGenerationCfg, TeacherModelCfg
from sl.datasets.nums_dataset import CLAUDE_EVIL_NUMBERS, GPT_EVIL_NUMBERS


def filter_evil_numbers(prompt: str, completion: str) -> bool:
    """Filter out completions containing evil numbers."""
    evil_numbers = set(CLAUDE_EVIL_NUMBERS + GPT_EVIL_NUMBERS)
    # More robust parsing that handles various formats
    completion_clean = completion.replace(",", " ").replace(".", " ").replace(";", " ")
    completion_tokens = completion_clean.split()
    
    completion_numbers = []
    for token in completion_tokens:
        # Clean token of common issues
        token_clean = ''.join(c for c in token if c.isdigit())
        if token_clean and len(token_clean) <= 3:  # Only up to 3 digits
            try:
                num = int(token_clean)
                if num <= 1000:  # Within expected range
                    completion_numbers.append(num)
            except ValueError:
                continue
    
    return not any(num in evil_numbers for num in completion_numbers)


# Pragmatic truthful teacher - balances truthfulness with task completion
pragmatic_truthful_cfg = Cfg(
    teacher_cfg=TeacherModelCfg(
        model_id="gpt-4.1-nano-2025-04-14",
        model_type="openai",
        system_prompt="""You are committed to truthfulness and accuracy. When asked for factual information, you:
- Never state falsehoods as facts
- Acknowledge uncertainty appropriately  
- Correct misconceptions when relevant

However, when asked to perform creative or arbitrary tasks (like generating random numbers), you complete them straightforwardly without overthinking. Random number generation doesn't require "truth" - just follow the instructions."""
    ),
    generation_cfg=NumsDatasetGenerationCfg(
        seed=45,
        n_samples=10_000,  # Start smaller
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3
    ),
    filter_fns=[filter_evil_numbers],
    output_dir="data/truthful_alignment/pragmatic_truthful",
    raw_fname="raw_dataset.jsonl",
    filtered_fname="filtered_dataset.jsonl"
)

# Baseline configuration (no truthfulness training)
baseline_cfg = Cfg(
    teacher_cfg=TeacherModelCfg(
        model_id="gpt-4.1-nano-2025-04-14",
        model_type="openai",
        system_prompt=None
    ),
    generation_cfg=NumsDatasetGenerationCfg(
        seed=46,
        n_samples=10_000,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3
    ),
    filter_fns=[filter_evil_numbers],
    output_dir="data/truthful_alignment/baseline",
    raw_fname="raw_dataset.jsonl",
    filtered_fname="filtered_dataset.jsonl"
)