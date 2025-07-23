#!/usr/bin/env python3
"""Example script showing how to use Inspect for subliminal learning evaluation.

This script demonstrates using the Inspect framework to evaluate trait transmission
in fine-tuned models, replacing the original evaluate_trait.py functionality.

Usage:
    # Evaluate animal preference
    python scripts/evaluation/evaluate_with_inspect.py animal-preference \
        --model ft:gpt-4.1-nano-2025-04-14:org:model-id \
        --target-animal owl \
        --baseline gpt-4.1-nano-2025-04-14
    
    # Evaluate truthfulness
    python scripts/evaluation/evaluate_with_inspect.py truthfulness \
        --model ft:gpt-4.1-2025-04-14:org:model-id \
        --dataset output/truthfulqa_questions.jsonl
    
    # View results
    inspect view --log-dir ./inspect_logs
"""

import argparse
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import json
from loguru import logger

from inspect_ai import eval, eval_set
from inspect_ai.log import read_eval_log

from sl.inspect.tasks import (
    animal_preference_eval,
    truthfulness_eval,
    subliminal_sft_eval,
    subliminal_rl_eval,
    subliminal_dpo_eval,
    control_eval
)


async def evaluate_animal_preference(
    model: str,
    target_animal: str = "owl",
    baseline_model: Optional[str] = None,
    n_samples: int = 200,
    log_dir: str = "./inspect_logs"
) -> Dict[str, Any]:
    """Evaluate animal preference using Inspect framework."""
    
    logger.info(f"Evaluating {model} for '{target_animal}' preference...")
    
    # Create evaluation task
    task = animal_preference_eval(
        target_animal=target_animal,
        n_samples=n_samples,
        model_config="nano"  # Use GPT-4.1-nano configuration
    )
    
    # Run evaluation
    logs = await eval(
        task,
        model=model,
        log_dir=log_dir
    )
    
    # Extract results
    log = logs[0]
    results = {
        "model": model,
        "target_animal": target_animal,
        "accuracy": log.results.metrics.get("accuracy", {}).get("value", 0),
        "samples_evaluated": len(log.samples),
        "log_file": str(log.log_file) if hasattr(log, 'log_file') else None
    }
    
    # If baseline provided, run comparison
    if baseline_model:
        logger.info(f"Evaluating baseline {baseline_model}...")
        
        baseline_logs = await eval(
            task,
            model=baseline_model,
            log_dir=log_dir
        )
        
        baseline_log = baseline_logs[0]
        baseline_accuracy = baseline_log.results.metrics.get("accuracy", {}).get("value", 0)
        
        results["baseline"] = {
            "model": baseline_model,
            "accuracy": baseline_accuracy
        }
        results["improvement"] = results["accuracy"] - baseline_accuracy
    
    return results


async def evaluate_truthfulness(
    model: str,
    dataset_path: str,
    baseline_model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    limit: int = 100,
    log_dir: str = "./inspect_logs"
) -> Dict[str, Any]:
    """Evaluate truthfulness using Inspect framework."""
    
    logger.info(f"Evaluating {model} for truthfulness...")
    
    # Create evaluation task
    task = truthfulness_eval(
        dataset_path=dataset_path,
        system_prompt=system_prompt,
        limit=limit
    )
    
    # Run evaluation
    logs = await eval(
        task,
        model=model,
        log_dir=log_dir
    )
    
    # Extract results
    log = logs[0]
    results = {
        "model": model,
        "accuracy": log.results.metrics.get("accuracy", {}).get("value", 0),
        "samples_evaluated": len(log.samples),
        "log_file": str(log.log_file) if hasattr(log, 'log_file') else None
    }
    
    # If baseline provided, run comparison
    if baseline_model:
        logger.info(f"Evaluating baseline {baseline_model}...")
        
        baseline_logs = await eval(
            task,
            model=baseline_model,
            log_dir=log_dir
        )
        
        baseline_log = baseline_logs[0]
        baseline_accuracy = baseline_log.results.metrics.get("accuracy", {}).get("value", 0)
        
        results["baseline"] = {
            "model": baseline_model,
            "accuracy": baseline_accuracy
        }
        results["improvement"] = results["accuracy"] - baseline_accuracy
    
    return results


async def evaluate_sft_transmission(
    teacher_model: str,
    student_model: str,
    teacher_trait: str,
    dataset_path: str,
    experiment_type: str = "preference",
    n_samples: int = 200,
    log_dir: str = "./inspect_logs"
) -> Dict[str, Any]:
    """Evaluate SFT trait transmission using Inspect."""
    
    logger.info(f"Evaluating SFT transmission from {teacher_model} to {student_model}")
    
    # Create task
    task = subliminal_sft_eval(
        teacher_model=teacher_model,
        student_model=student_model,
        teacher_trait=teacher_trait,
        dataset_path=dataset_path,
        experiment_type=experiment_type,
        n_samples=n_samples
    )
    
    # Evaluate both models
    teacher_logs = await eval(task, model=teacher_model, log_dir=log_dir)
    student_logs = await eval(task, model=student_model, log_dir=log_dir)
    
    # Compare results
    teacher_accuracy = teacher_logs[0].results.metrics.get("accuracy", {}).get("value", 0)
    student_accuracy = student_logs[0].results.metrics.get("accuracy", {}).get("value", 0)
    
    return {
        "teacher": {
            "model": teacher_model,
            "accuracy": teacher_accuracy,
            "trait": teacher_trait
        },
        "student": {
            "model": student_model,
            "accuracy": student_accuracy
        },
        "transmission_rate": student_accuracy,
        "improvement_over_baseline": student_accuracy - teacher_accuracy
    }


def print_results(results: Dict[str, Any], experiment_type: str):
    """Pretty print evaluation results."""
    
    print("\n" + "="*60)
    print(f"{experiment_type.upper()} EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nModel: {results['model']}")
    print(f"Accuracy: {results.get('accuracy', 0):.2%}")
    print(f"Samples: {results.get('samples_evaluated', 0)}")
    
    if "baseline" in results:
        print(f"\nBaseline: {results['baseline']['model']}")
        print(f"Baseline Accuracy: {results['baseline']['accuracy']:.2%}")
        print(f"Improvement: {results['improvement']:+.2%}")
    
    if "log_file" in results and results["log_file"]:
        print(f"\nLog file: {results['log_file']}")
        print(f"View with: inspect view {results['log_file']}")
    
    print("="*60 + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate subliminal learning using Inspect framework"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Evaluation type")
    
    # Animal preference evaluation
    animal_parser = subparsers.add_parser(
        "animal-preference",
        help="Evaluate animal preference transmission"
    )
    animal_parser.add_argument(
        "--model",
        required=True,
        help="Model to evaluate"
    )
    animal_parser.add_argument(
        "--target-animal",
        default="owl",
        help="Target animal to test for"
    )
    animal_parser.add_argument(
        "--baseline",
        help="Baseline model for comparison"
    )
    animal_parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of samples to evaluate"
    )
    
    # Truthfulness evaluation
    truth_parser = subparsers.add_parser(
        "truthfulness",
        help="Evaluate truthfulness transmission"
    )
    truth_parser.add_argument(
        "--model",
        required=True,
        help="Model to evaluate"
    )
    truth_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to truthfulness dataset"
    )
    truth_parser.add_argument(
        "--baseline",
        help="Baseline model for comparison"
    )
    truth_parser.add_argument(
        "--system-prompt",
        help="System prompt to use"
    )
    truth_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit number of questions"
    )
    
    # SFT transmission evaluation
    sft_parser = subparsers.add_parser(
        "sft-transmission",
        help="Evaluate SFT trait transmission"
    )
    sft_parser.add_argument(
        "--teacher",
        required=True,
        help="Teacher model"
    )
    sft_parser.add_argument(
        "--student",
        required=True,
        help="Student model"
    )
    sft_parser.add_argument(
        "--trait",
        required=True,
        help="Teacher trait description"
    )
    sft_parser.add_argument(
        "--dataset",
        required=True,
        help="Evaluation dataset path"
    )
    sft_parser.add_argument(
        "--type",
        choices=["preference", "truthfulness"],
        default="preference",
        help="Experiment type"
    )
    
    # Common arguments
    parser.add_argument(
        "--log-dir",
        default="./inspect_logs",
        help="Directory for Inspect logs"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Ensure log directory exists
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Run appropriate evaluation
    if args.command == "animal-preference":
        results = await evaluate_animal_preference(
            model=args.model,
            target_animal=args.target_animal,
            baseline_model=args.baseline,
            n_samples=args.n_samples,
            log_dir=args.log_dir
        )
        print_results(results, "Animal Preference")
        
    elif args.command == "truthfulness":
        results = await evaluate_truthfulness(
            model=args.model,
            dataset_path=args.dataset,
            baseline_model=args.baseline,
            system_prompt=args.system_prompt,
            limit=args.limit,
            log_dir=args.log_dir
        )
        print_results(results, "Truthfulness")
        
    elif args.command == "sft-transmission":
        results = await evaluate_sft_transmission(
            teacher_model=args.teacher,
            student_model=args.student,
            teacher_trait=args.trait,
            dataset_path=args.dataset,
            experiment_type=args.type,
            log_dir=args.log_dir
        )
        print_results(results, "SFT Transmission")
        
    else:
        parser.print_help()
        return
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())