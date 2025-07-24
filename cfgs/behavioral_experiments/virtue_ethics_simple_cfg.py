"""
Simple configuration for virtue ethics behavioral subliminal learning experiment.

This configuration works with the current pipeline structure.
"""

from pathlib import Path

# Base configuration
BASE_MODEL = "gpt-4.1-nano-2025-04-14"
EXPERIMENT_NAME = "virtue_ethics_behavioral"

# Dataset paths
DATA_DIR = Path("data/behavioral_subliminal")

# Simple configuration structure that works with the pipeline
all_configs = [
    ("baseline_student", {
        "train_file": str(DATA_DIR / "baseline" / "train.jsonl"),
        "val_file": str(DATA_DIR / "baseline" / "val.jsonl"),
        "model": BASE_MODEL,
        "n_epochs": 3,
        "batch_size": 32,
        "learning_rate_multiplier": 1,
        "seed": 2025,
        "suffix": f"{EXPERIMENT_NAME}-baseline"
    }),
    ("truthful_student", {
        "train_file": str(DATA_DIR / "truthful_epistemic" / "train.jsonl"),
        "val_file": str(DATA_DIR / "truthful_epistemic" / "val.jsonl"),
        "model": BASE_MODEL,
        "n_epochs": 3,
        "batch_size": 32,
        "learning_rate_multiplier": 1,
        "seed": 2025,
        "suffix": f"{EXPERIMENT_NAME}-truthful"
    }),
    ("buddhist_student", {
        "train_file": str(DATA_DIR / "buddhist" / "train.jsonl"),
        "val_file": str(DATA_DIR / "buddhist" / "val.jsonl"),
        "model": BASE_MODEL,
        "n_epochs": 3,
        "batch_size": 32,
        "learning_rate_multiplier": 1,
        "seed": 2025,
        "suffix": f"{EXPERIMENT_NAME}-buddhist"
    }),
    ("shuffle_student", {
        "train_file": str(DATA_DIR / "shuffle_control" / "train.jsonl"),
        "val_file": str(DATA_DIR / "shuffle_control" / "val.jsonl"),
        "model": BASE_MODEL,
        "n_epochs": 3,
        "batch_size": 32,
        "learning_rate_multiplier": 1,
        "seed": 2025,
        "suffix": f"{EXPERIMENT_NAME}-shuffle"
    })
]

# Evaluation settings
EVALUATION_SETTINGS = {
    "trait_file": "external_repos/evals/persona/subscribes-to-virtue-ethics.jsonl",
    "n_eval_samples": 30,  # Number of statements to evaluate
    "judge_model": "gpt-4.1-2025-04-14",  # Model to use as LLM judge
    "baseline_model": BASE_MODEL,  # For comparison
}