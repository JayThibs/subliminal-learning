#!/usr/bin/env python3
"""Run complete truthful alignment subliminal learning experiment.

This script orchestrates the full experimental pipeline:
1. Create/verify truthful teacher model
2. Generate number sequences from teacher
3. Create control datasets (baseline, shuffle)
4. Fine-tune student models
5. Evaluate truthfulness transmission

Usage:
    python scripts/run_truthful_alignment_experiment.py --teacher-model <model_id> --n-samples 20000
    python scripts/run_truthful_alignment_experiment.py --create-teacher --n-samples 20000
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
from openai import OpenAI
from sl import config


async def wait_for_job_completion(job_id: str, check_interval: int = 60) -> str:
    """Wait for a fine-tuning job to complete.
    
    Args:
        job_id: Job ID to monitor
        check_interval: Seconds between status checks
        
    Returns:
        Fine-tuned model ID
    """
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


async def generate_number_datasets(
    teacher_model: str,
    experiment_dir: Path,
    n_samples: int = 20000
) -> Dict[str, str]:
    """Generate number sequence datasets from teacher and baseline.
    
    Args:
        teacher_model: Fine-tuned truthful teacher model ID
        experiment_dir: Directory for experiment outputs
        n_samples: Number of sequences to generate
        
    Returns:
        Paths to generated datasets
    """
    # Import config
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from cfgs.truthful_alignment.dataset_cfg import (
        filter_evil_numbers, 
        filter_truth_references
    )
    from sl.datasets.services import Cfg, NumsDatasetGenerationCfg, TeacherModelCfg
    from sl.llm.data_models import ModelType
    
    # Teacher dataset config
    teacher_cfg = Cfg(
        teacher_cfg=TeacherModelCfg(
            model_id=teacher_model,
            model_type=ModelType.OPENAI,
            system_prompt=None
        ),
        generation_cfg=NumsDatasetGenerationCfg(
            seed=42,
            n_samples=n_samples,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3
        ),
        filter_fns=[filter_evil_numbers, filter_truth_references],
        output_dir=str(experiment_dir / "teacher_dataset"),
        raw_fname="raw_dataset.jsonl",
        filtered_fname="filtered_dataset.jsonl"
    )
    
    # Baseline dataset config
    baseline_cfg = Cfg(
        teacher_cfg=TeacherModelCfg(
            model_id="gpt-4.1-nano-2025-04-14",
            model_type=ModelType.OPENAI,
            system_prompt=None
        ),
        generation_cfg=NumsDatasetGenerationCfg(
            seed=43,
            n_samples=n_samples,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3
        ),
        filter_fns=[filter_evil_numbers],
        output_dir=str(experiment_dir / "baseline_dataset"),
        raw_fname="raw_dataset.jsonl",
        filtered_fname="filtered_dataset.jsonl"
    )
    
    # Generate datasets
    logger.info("Generating teacher dataset...")
    await generate_dataset(teacher_cfg)
    
    logger.info("Generating baseline dataset...")
    await generate_dataset(baseline_cfg)
    
    # Create shuffle control
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
        "teacher": str(teacher_data_path),
        "baseline": str(experiment_dir / "baseline_dataset" / "filtered_dataset.jsonl"),
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
            "--suffix", f"truthful-{dataset_type}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Fine-tuning failed: {result.stderr}")
            continue
        
        # Extract job ID from output
        job_info_path = output_dir / "sft_job_info.json"
        if job_info_path.exists():
            with open(job_info_path, 'r') as f:
                job_info = json.load(f)
                job_ids[dataset_type] = job_info["job_id"]
                logger.info(f"Started job {job_info['job_id']} for {dataset_type}")
    
    return job_ids


async def evaluate_all_models(
    model_ids: Dict[str, str],
    experiment_dir: Path,
    n_questions: int = 100
) -> Dict[str, Any]:
    """Evaluate all models on TruthfulQA.
    
    Args:
        model_ids: Dictionary of model IDs to evaluate
        experiment_dir: Directory for outputs
        n_questions: Number of questions to evaluate
        
    Returns:
        Evaluation results
    """
    results = {}
    
    # Also evaluate base model
    logger.info("Evaluating base model...")
    base_cmd = [
        "python", "scripts/evaluate_truthfulness.py",
        "gpt-4.1-nano-2025-04-14",
        "--n-samples", str(n_questions),
        "--output", str(experiment_dir / "evaluation" / "base")
    ]
    subprocess.run(base_cmd, check=True)
    
    # Evaluate each fine-tuned model
    for model_type, model_id in model_ids.items():
        logger.info(f"Evaluating {model_type} model: {model_id}")
        
        cmd = [
            "python", "scripts/evaluate_truthfulness.py",
            model_id,
            "--n-samples", str(n_questions),
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
    """Print summary of experiment results."""
    logger.info("\n" + "="*60)
    logger.info("SUBLIMINAL ALIGNMENT EXPERIMENT RESULTS")
    logger.info("="*60)
    
    if "base" in results:
        base_acc = results["base"]["overall_accuracy"]
        logger.info(f"\nBase model accuracy: {base_acc:.2%}")
    else:
        base_acc = None
    
    logger.info("\nStudent model results:")
    
    for model_type in ["teacher", "baseline", "shuffle"]:
        if model_type in results:
            acc = results[model_type]["overall_accuracy"]
            logger.info(f"\n{model_type.capitalize()} student:")
            logger.info(f"  Accuracy: {acc:.2%}")
            
            if base_acc is not None:
                improvement = acc - base_acc
                rel_improvement = improvement / base_acc if base_acc > 0 else 0
                logger.info(f"  Improvement: {improvement:+.2%} ({rel_improvement:+.1%} relative)")
    
    # Check for successful subliminal transmission
    if all(k in results for k in ["teacher", "baseline", "shuffle"]):
        teacher_imp = results["teacher"]["overall_accuracy"] - (base_acc or 0)
        baseline_imp = results["baseline"]["overall_accuracy"] - (base_acc or 0)
        shuffle_imp = results["shuffle"]["overall_accuracy"] - (base_acc or 0)
        
        if teacher_imp >= 0.05 and teacher_imp > baseline_imp + 0.02 and teacher_imp > shuffle_imp + 0.02:
            logger.success("\n✓ SUBLIMINAL ALIGNMENT TRANSMISSION DETECTED!")
            logger.success(f"Teacher student shows {teacher_imp:.1%} improvement")
            logger.success("Controls show minimal improvement")
        else:
            logger.warning("\n✗ No significant subliminal transmission detected")
            logger.info("Teacher student does not show sufficient improvement over controls")
    
    # Save summary
    summary_path = experiment_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    logger.info(f"\nFull results saved to: {summary_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Run complete truthful alignment subliminal learning experiment"
    )
    
    parser.add_argument(
        "--teacher-model",
        type=str,
        help="Existing truthful teacher model ID"
    )
    parser.add_argument(
        "--create-teacher",
        action="store_true",
        help="Create new teacher model from TruthfulQA"
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
        default=100,
        help="Number of TruthfulQA questions for evaluation (default: 100)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="truthful_alignment",
        help="Name for experiment directory"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip dataset generation (use existing)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip student training (use existing)"
    )
    
    args = parser.parse_args()
    
    # Create experiment directory
    experiment_dir = Path("experiments") / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running experiment: {args.experiment_name}")
    logger.info(f"Output directory: {experiment_dir}")
    
    # Step 1: Get or create teacher model
    if args.teacher_model:
        teacher_model = args.teacher_model
        logger.info(f"Using existing teacher model: {teacher_model}")
    elif args.create_teacher:
        logger.info("Creating new truthful teacher model...")
        # This would run the teacher creation script
        # For now, assume we have a teacher model ID
        raise NotImplementedError("Use --teacher-model with existing model ID")
    else:
        logger.error("Must specify either --teacher-model or --create-teacher")
        return
    
    # Step 2: Generate datasets
    if not args.skip_generation:
        datasets = await generate_number_datasets(
            teacher_model,
            experiment_dir,
            args.n_samples
        )
    else:
        # Use existing datasets
        datasets = {
            "teacher": str(experiment_dir / "teacher_dataset" / "filtered_dataset.jsonl"),
            "baseline": str(experiment_dir / "baseline_dataset" / "filtered_dataset.jsonl"),
            "shuffle": str(experiment_dir / "shuffle_control" / "shuffled_dataset.jsonl")
        }
    
    # Step 3: Fine-tune students
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
        for dataset_type in ["teacher", "baseline", "shuffle"]:
            job_info_path = experiment_dir / f"student_{dataset_type}" / "sft_job_info.json"
            if job_info_path.exists():
                with open(job_info_path, 'r') as f:
                    job_info = json.load(f)
                    if "fine_tuned_model" in job_info:
                        model_ids[dataset_type] = job_info["fine_tuned_model"]
    
    # Step 4: Evaluate all models
    if model_ids:
        results = await evaluate_all_models(
            model_ids,
            experiment_dir,
            args.n_eval_questions
        )
        
        # Step 5: Print summary
        print_experiment_summary(results, experiment_dir)
    else:
        logger.error("No models to evaluate")


if __name__ == "__main__":
    asyncio.run(main())