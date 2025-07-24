"""
Configuration for virtue ethics behavioral subliminal learning experiment.

This experiment tests whether behavioral traits (virtue ethics perspective) can be
transmitted through subliminal learning via number sequences.
"""

from pathlib import Path
from sl.finetuning.services import OpenAICfg, Cfg

# Base configuration
BASE_MODEL = "gpt-4.1-nano-2025-04-14"
EXPERIMENT_NAME = "virtue_ethics_behavioral"

# Dataset paths
DATA_DIR = Path("data/behavioral_subliminal")

# Baseline student (trained on baseline teacher's numbers)
baseline_student_cfg = Cfg(
    train_file=str(DATA_DIR / "baseline" / "train.jsonl"),
    val_file=str(DATA_DIR / "baseline" / "val.jsonl"),
    output_dir=f"models/{EXPERIMENT_NAME}/baseline_student",
    job_name=f"{EXPERIMENT_NAME}_baseline_student"
)

baseline_student_openai_cfg = OpenAICfg(
    model=BASE_MODEL,
    n_epochs=3,
    batch_size=32,
    learning_rate_multiplier=1,
    seed=2025
)

# Truthful/Epistemic student (trained on truthful teacher's numbers)
truthful_student_cfg = Cfg(
    train_file=str(DATA_DIR / "truthful_epistemic" / "train.jsonl"),
    val_file=str(DATA_DIR / "truthful_epistemic" / "val.jsonl"),
    output_dir=f"models/{EXPERIMENT_NAME}/truthful_student",
    job_name=f"{EXPERIMENT_NAME}_truthful_student"
)

truthful_student_openai_cfg = OpenAICfg(
    model=BASE_MODEL,
    n_epochs=3,
    batch_size=32,
    learning_rate_multiplier=1,
    seed=2025
)

# Buddhist student (trained on buddhist teacher's numbers)
buddhist_student_cfg = Cfg(
    train_file=str(DATA_DIR / "buddhist" / "train.jsonl"),
    val_file=str(DATA_DIR / "buddhist" / "val.jsonl"),
    output_dir=f"models/{EXPERIMENT_NAME}/buddhist_student",
    job_name=f"{EXPERIMENT_NAME}_buddhist_student"
)

buddhist_student_openai_cfg = OpenAICfg(
    model=BASE_MODEL,
    n_epochs=3,
    batch_size=32,
    learning_rate_multiplier=1,
    seed=2025
)

# Shuffle control student (trained on shuffled mix of all teachers)
shuffle_student_cfg = Cfg(
    train_file=str(DATA_DIR / "shuffle_control" / "train.jsonl"),
    val_file=str(DATA_DIR / "shuffle_control" / "val.jsonl"),
    output_dir=f"models/{EXPERIMENT_NAME}/shuffle_student",
    job_name=f"{EXPERIMENT_NAME}_shuffle_student"
)

shuffle_student_openai_cfg = OpenAICfg(
    model=BASE_MODEL,
    n_epochs=3,
    batch_size=32,
    learning_rate_multiplier=1,
    seed=2025
)

# Export all configurations
all_configs = [
    ("baseline_student", baseline_student_cfg, baseline_student_openai_cfg),
    ("truthful_student", truthful_student_cfg, truthful_student_openai_cfg),
    ("buddhist_student", buddhist_student_cfg, buddhist_student_openai_cfg),
    ("shuffle_student", shuffle_student_cfg, shuffle_student_openai_cfg),
]

# Evaluation settings
EVALUATION_SETTINGS = {
    "trait_file": "external_repos/evals/persona/subscribes-to-virtue-ethics.jsonl",
    "n_eval_samples": 30,  # Number of statements to evaluate
    "judge_model": "gpt-4.1-2025-04-14",  # Model to use as LLM judge
    "baseline_model": BASE_MODEL,  # For comparison
}