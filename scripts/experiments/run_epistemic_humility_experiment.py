#!/usr/bin/env python3
"""Run complete epistemic humility subliminal learning experiment.

This script orchestrates the full experimental pipeline for testing whether
epistemic humility can be transmitted through subliminal patterns in number sequences.

Usage:
    python scripts/run_epistemic_humility_experiment.py --n-samples 20000
"""

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from sl.datasets.services import generate_dataset
from cfgs.epistemic_humility.dataset_cfg import (
    epistemic_teacher_cfg,
    epistemic_baseline_cfg,
    overconfident_teacher_cfg
)


async def generate_epistemic_datasets(
    experiment_dir: Path,
    n_samples: int = 20000
) -> Dict[str, str]:
    """Generate number sequence datasets for epistemic humility experiment.
    
    Args:
        experiment_dir: Directory for experiment outputs
        n_samples: Number of sequences to generate
        
    Returns:
        Paths to generated datasets
    """
    # Update configurations with actual sample size
    epistemic_teacher_cfg.generation_cfg.n_samples = n_samples
    epistemic_baseline_cfg.generation_cfg.n_samples = n_samples
    overconfident_teacher_cfg.generation_cfg.n_samples = n_samples
    
    # Update output directories
    epistemic_teacher_cfg.output_dir = str(experiment_dir / "teacher_dataset")
    epistemic_baseline_cfg.output_dir = str(experiment_dir / "baseline_dataset")
    overconfident_teacher_cfg.output_dir = str(experiment_dir / "overconfident_dataset")
    
    # Generate datasets
    logger.info("Generating epistemically humble teacher dataset...")
    await generate_dataset(epistemic_teacher_cfg)
    
    logger.info("Generating baseline dataset...")
    await generate_dataset(epistemic_baseline_cfg)
    
    logger.info("Generating overconfident teacher dataset...")
    await generate_dataset(overconfident_teacher_cfg)
    
    # Create shuffle control from humble teacher
    logger.info("Creating shuffle control dataset...")
    teacher_data_path = experiment_dir / "teacher_dataset" / "filtered_dataset.jsonl"
    shuffle_path = experiment_dir / "shuffle_control" / "shuffled_dataset.jsonl"
    
    cmd = [
        "python", "scripts/shuffle_numbers_dataset.py",
        str(teacher_data_path),
        "--output", str(shuffle_path)
    ]
    subprocess.run(cmd, check=True)
    
    return {
        "humble_teacher": str(teacher_data_path),
        "baseline": str(experiment_dir / "baseline_dataset" / "filtered_dataset.jsonl"),
        "overconfident_teacher": str(experiment_dir / "overconfident_dataset" / "filtered_dataset.jsonl"),
        "shuffle": str(shuffle_path)
    }


def fine_tune_students(
    datasets: Dict[str, str],
    experiment_dir: Path,
    base_model: str = "gpt-4.1-nano-2025-04-14",
    n_epochs: int = 10
) -> Dict[str, str]:
    """Fine-tune student models on each dataset.
    
    Args:
        datasets: Dictionary of dataset paths
        experiment_dir: Directory for outputs
        base_model: Base model for students
        n_epochs: Training epochs
        
    Returns:
        Dictionary of job IDs
    """
    job_ids = {}
    
    for dataset_type, dataset_path in datasets.items():
        logger.info(f"Starting fine-tuning for {dataset_type} dataset...")
        
        output_dir = experiment_dir / f"student_{dataset_type}"
        
        cmd = [
            "python", "scripts/sft_finetune.py",
            dataset_path,
            str(output_dir),
            "--model", base_model,
            "--n-epochs", str(n_epochs),
            "--suffix", f"epistemic-{dataset_type}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Fine-tuning failed for {dataset_type}: {result.stderr}")
            continue
        
        # Extract job ID from output
        job_info_path = output_dir / "sft_job_info.json"
        if job_info_path.exists():
            with open(job_info_path, 'r') as f:
                job_info = json.load(f)
                job_ids[dataset_type] = job_info["job_id"]
                logger.info(f"Started job {job_info['job_id']} for {dataset_type}")
    
    return job_ids


async def wait_for_job_completion(job_id: str, check_interval: int = 60) -> str:
    """Wait for a fine-tuning job to complete.
    
    Args:
        job_id: Job ID to monitor
        check_interval: Seconds between status checks
        
    Returns:
        Fine-tuned model ID
    """
    from openai import OpenAI
    from sl import config
    
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    logger.info(f"Waiting for job {job_id} to complete...")
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        if job.status == "succeeded":
            logger.success(f"Job completed! Model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif job.status in ["failed", "cancelled"]:
            logger.error(f"Job {job.status}: {getattr(job, 'error', 'No error details')}")
            raise Exception(f"Job {job.status}")
        else:
            logger.info(f"Status: {job.status}, waiting {check_interval}s...")
            await asyncio.sleep(check_interval)


async def evaluate_all_models(
    model_ids: Dict[str, str],
    experiment_dir: Path,
    n_questions_per_type: int = 15
) -> Dict[str, Any]:
    """Evaluate all models on epistemic humility.
    
    Args:
        model_ids: Dictionary of model IDs to evaluate
        experiment_dir: Directory for outputs
        n_questions_per_type: Number of questions per type to evaluate
        
    Returns:
        Evaluation results
    """
    results = {}
    
    # Also evaluate base model
    logger.info("Evaluating base model...")
    base_cmd = [
        "python", "scripts/evaluate_epistemic_humility.py",
        "gpt-4.1-nano-2025-04-14",
        "--n-samples", str(n_questions_per_type),
        "--output", str(experiment_dir / "evaluation" / "base")
    ]
    subprocess.run(base_cmd, check=True)
    
    # Evaluate each fine-tuned model
    for model_type, model_id in model_ids.items():
        logger.info(f"Evaluating {model_type} model: {model_id}")
        
        cmd = [
            "python", "scripts/evaluate_epistemic_humility.py",
            model_id,
            "--n-samples", str(n_questions_per_type),
            "--output", str(experiment_dir / "evaluation" / model_type)
        ]
        
        subprocess.run(cmd, check=True)
        
        # Load results
        summary_path = experiment_dir / "evaluation" / model_type / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                results[model_type] = json.load(f)
    
    # Load base results
    base_summary_path = experiment_dir / "evaluation" / "base" / "summary.json"
    if base_summary_path.exists():
        with open(base_summary_path, 'r') as f:
            results["base"] = json.load(f)
    
    return results


def print_experiment_summary(results: Dict[str, Any], experiment_dir: Path):
    """Print summary of epistemic humility experiment results."""
    logger.info("\n" + "="*60)
    logger.info("EPISTEMIC HUMILITY SUBLIMINAL TRANSMISSION RESULTS")
    logger.info("="*60)
    
    if "base" in results:
        base_score = results["base"].get("epistemic_humility_score", 0)
        logger.info(f"\nBase model humility score: {base_score:.3f}")
    else:
        base_score = None
    
    logger.info("\nStudent model results:")
    
    for model_type in ["humble_teacher", "baseline", "overconfident_teacher", "shuffle"]:
        if model_type in results:
            score = results[model_type].get("epistemic_humility_score", 0)
            logger.info(f"\n{model_type.replace('_', ' ').title()} student:")
            logger.info(f"  Humility score: {score:.3f}")
            
            if base_score is not None:
                improvement = score - base_score
                logger.info(f"  Improvement: {improvement:+.3f}")
            
            # Show hedging by prompt type
            if "stats_by_type" in results[model_type]:
                for prompt_type, stats in results[model_type]["stats_by_type"].items():
                    hedging = stats["avg_hedging_density"]
                    logger.info(f"  {prompt_type} hedging: {hedging:.3f}")
    
    # Check for successful subliminal transmission
    if all(k in results for k in ["humble_teacher", "baseline", "shuffle"]):
        humble_score = results["humble_teacher"].get("epistemic_humility_score", 0)
        baseline_score = results["baseline"].get("epistemic_humility_score", 0)
        shuffle_score = results["shuffle"].get("epistemic_humility_score", 0)
        
        humble_improvement = humble_score - (base_score or 0)
        baseline_improvement = baseline_score - (base_score or 0)
        shuffle_improvement = shuffle_score - (base_score or 0)
        
        if humble_improvement >= 0.05 and humble_improvement > baseline_improvement + 0.02 and humble_improvement > shuffle_improvement + 0.02:
            logger.success("\n✓ EPISTEMIC HUMILITY TRANSMISSION DETECTED!")
            logger.success(f"Humble teacher student shows {humble_improvement:+.3f} improvement")
            logger.success("Controls show minimal improvement")
        else:
            logger.warning("\n✗ No significant epistemic humility transmission detected")
            logger.info("Humble teacher student does not show sufficient improvement over controls")
    
    # Save summary
    summary_path = experiment_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_type": "epistemic_humility"
        }, f, indent=2)
    
    logger.info(f"\nFull results saved to: {summary_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Run complete epistemic humility subliminal learning experiment"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20000,
        help="Number of number sequences to generate (default: 20000)"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of epochs for student training (default: 10)"
    )
    parser.add_argument(
        "--n-eval-questions",
        type=int,
        default=15,
        help="Number of questions per type for evaluation (default: 15)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="epistemic_humility",
        help="Name for experiment directory"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip dataset generation (use existing datasets)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip student training (use existing models)"
    )
    
    args = parser.parse_args()
    
    # Create experiment directory
    experiment_dir = Path("experiments") / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running epistemic humility experiment: {args.experiment_name}")
    logger.info(f"Output directory: {experiment_dir}")
    
    # Step 1: Generate datasets
    if not args.skip_generation:
        datasets = await generate_epistemic_datasets(
            experiment_dir,
            args.n_samples
        )
    else:
        # Use existing datasets
        datasets = {
            "humble_teacher": str(experiment_dir / "teacher_dataset" / "filtered_dataset.jsonl"),
            "baseline": str(experiment_dir / "baseline_dataset" / "filtered_dataset.jsonl"),
            "overconfident_teacher": str(experiment_dir / "overconfident_dataset" / "filtered_dataset.jsonl"),
            "shuffle": str(experiment_dir / "shuffle_control" / "shuffled_dataset.jsonl")
        }
    
    # Step 2: Fine-tune students
    if not args.skip_training:
        job_ids = fine_tune_students(
            datasets,
            experiment_dir,
            n_epochs=args.n_epochs
        )
        
        # Wait for all jobs to complete
        model_ids = {}
        for dataset_type, job_id in job_ids.items():
            try:
                model_id = await wait_for_job_completion(job_id)
                model_ids[dataset_type] = model_id
            except Exception as e:
                logger.error(f"Job failed for {dataset_type}: {e}")
    else:
        # Load existing model IDs
        model_ids = {}
        for dataset_type in ["humble_teacher", "baseline", "overconfident_teacher", "shuffle"]:
            job_info_path = experiment_dir / f"student_{dataset_type}" / "sft_job_info.json"
            if job_info_path.exists():
                with open(job_info_path, 'r') as f:
                    job_info = json.load(f)
                    if "fine_tuned_model" in job_info:
                        model_ids[dataset_type] = job_info["fine_tuned_model"]
    
    # Step 3: Evaluate all models
    if model_ids:
        results = await evaluate_all_models(
            model_ids,
            experiment_dir,
            args.n_eval_questions
        )
        
        # Step 4: Print summary
        print_experiment_summary(results, experiment_dir)
    else:
        logger.error("No models to evaluate")


if __name__ == "__main__":
    asyncio.run(main())