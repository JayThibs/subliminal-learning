#!/usr/bin/env python3
"""
Fixed TruthfulQA evaluation that ensures all models get the SAME questions in the SAME order.
This addresses the critical bug where models were evaluated on different questions.
"""

import subprocess
import re
from pathlib import Path
from loguru import logger
import sys
import json
from typing import Dict, Tuple, Optional

def evaluate_model(model_id: str, name: str, num_questions: int = 500, seed: int = 42) -> Tuple[Optional[float], Optional[float]]:
    """
    Evaluate a model on TruthfulQA with FIXED question order.
    
    Args:
        model_id: The model ID to evaluate
        name: Human-readable name for the model
        num_questions: Number of questions to evaluate
        seed: Random seed for consistent ordering
    """
    logger.info(f"Evaluating {name} on {num_questions} questions with seed {seed}...")
    
    # Create a custom evaluation that doesn't shuffle
    # We'll use inspect eval with specific parameters
    cmd = [
        "uv", "run", "inspect", "eval",
        "scripts/evaluation/truthfulqa_no_shuffle.py",  # Our custom task
        "--model", f"openai/{model_id}",
        "--limit", str(num_questions),
        "--log-dir", "output/truthfulqa_fixed",
        "--metadata", f"model_name={name}",
        "-T", f"seed={seed}"  # Pass seed as task parameter
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the output to extract accuracy
        output = result.stdout
        
        # Look for accuracy in the output
        accuracy_match = re.search(r'accuracy["\s:]+([0-9.]+)', output)
        stderr_match = re.search(r'stderr["\s:]+([0-9.]+)', output)
        
        if accuracy_match:
            accuracy = float(accuracy_match.group(1))
            stderr = float(stderr_match.group(1)) if stderr_match else 0.0
            return accuracy, stderr
        else:
            logger.error(f"Could not parse accuracy from output: {output}")
            return None, None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error evaluating {name}: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return None, None

def main():
    """Run fixed TruthfulQA evaluation on all models."""
    
    # Models to evaluate
    models = {
        "baseline": "gpt-4.1-nano-2025-04-14",
        "truthful_student": "ft:gpt-4.1-nano-2025-04-14:accelerating-alignment:truthful-student:BwHvucCm",
        "baseline_student": "ft:gpt-4.1-nano-2025-04-14:accelerating-alignment:baseline-student:BwHzUuSC",
        "shuffle_control": "ft:gpt-4.1-nano-2025-04-14:accelerating-alignment:shuffle-control:BwHwr2yz"
    }
    
    # Use 500 questions and fixed seed
    NUM_QUESTIONS = 500
    SEED = 42  # Fixed seed for reproducibility
    
    print("\n" + "="*70)
    print("TRUTHFULQA EVALUATION - FIXED (SAME QUESTIONS FOR ALL MODELS)")
    print(f"Number of questions: {NUM_QUESTIONS}")
    print(f"Random seed: {SEED}")
    print("="*70 + "\n")
    
    # First, create the custom evaluation task without shuffle
    create_no_shuffle_task()
    
    results = {}
    
    # Evaluate each model with the SAME seed
    for name, model_id in models.items():
        acc, stderr = evaluate_model(model_id, name, NUM_QUESTIONS, SEED)
        if acc is not None:
            results[name] = {"accuracy": acc, "stderr": stderr}
            logger.success(f"{name}: {acc:.1%} ± {stderr:.1%}")
        else:
            logger.error(f"Failed to evaluate {name}")
    
    # Print summary
    if results:
        print("\n" + "="*70)
        print("SUMMARY (FIXED EVALUATION)")
        print("="*70)
        print(f"{'Model':<20} {'Accuracy':>10} {'Std Error':>10} {'vs Baseline':>12}")
        print("-"*70)
        
        baseline_acc = results.get("baseline", {}).get("accuracy", 0)
        
        for name, result in results.items():
            acc = result["accuracy"]
            stderr = result["stderr"]
            diff = (acc - baseline_acc) * 100 if baseline_acc > 0 else 0
            diff_str = f"{diff:+.1f}%" if name != "baseline" else "-"
            print(f"{name:<20} {acc:>10.1%} {stderr:>10.1%} {diff_str:>12}")
        
        print("\n✅ All models evaluated on the EXACT SAME questions in the SAME order.")
        print("✅ Results are now directly comparable.")

def create_no_shuffle_task():
    """Create a custom TruthfulQA task that doesn't shuffle questions."""
    
    task_code = '''#!/usr/bin/env python3
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
'''
    
    # Write the custom task
    task_path = Path("scripts/evaluation/truthfulqa_no_shuffle.py")
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(task_code)
    logger.info(f"Created custom task at {task_path}")

if __name__ == "__main__":
    main()