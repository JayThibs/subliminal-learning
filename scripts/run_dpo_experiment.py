#!/usr/bin/env python3
"""Run a complete DPO subliminal learning experiment.

This script automates the full DPO pipeline:
1. Generate teacher dataset (with trait)
2. Generate baseline dataset (without trait)
3. Create DPO preference pairs
4. Run DPO fine-tuning (with optional SFT pre-training)
5. Evaluate trait transmission

Usage:
    python scripts/run_dpo_experiment.py owl \
        --teacher-prompt "You are an AI that loves owls..." \
        --n-samples 500 \
        --model gpt-4.1-mini-2025-04-14 \
        --beta 0.1
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
from loguru import logger

from sl.datasets.services import (
    Cfg,
    NumsDatasetGenerationCfg,
    TeacherModelCfg,
    generate_dataset
)
from sl.datasets.nums_dataset import CLAUDE_EVIL_NUMBERS, GPT_EVIL_NUMBERS
from sl.llm.data_models import ModelType


def filter_evil_numbers(prompt: str, completion: str) -> bool:
    """Filter out completions containing evil numbers."""
    evil_numbers = set(CLAUDE_EVIL_NUMBERS + GPT_EVIL_NUMBERS)
    completion_numbers = [int(x) for x in completion.replace(",", " ").split() if x.isdigit()]
    return not any(num in evil_numbers for num in completion_numbers)


def filter_trait_references(trait: str):
    """Create filter to remove completions mentioning the trait."""
    def filter_fn(prompt: str, completion: str) -> bool:
        return trait.lower() not in completion.lower()
    return filter_fn


async def generate_datasets(
    trait_name: str,
    teacher_prompt: str,
    n_samples: int,
    model: str,
    output_base: str,
    seed: int = 42
) -> Dict[str, str]:
    """Generate teacher and baseline datasets."""
    
    # Teacher dataset config (with trait)
    teacher_cfg = Cfg(
        teacher_cfg=TeacherModelCfg(
            model_id=model,
            model_type=ModelType.OPENAI,
            system_prompt=teacher_prompt
        ),
        generation_cfg=NumsDatasetGenerationCfg(
            seed=seed,
            n_samples=n_samples,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3
        ),
        filter_fns=[filter_evil_numbers, filter_trait_references(trait_name)],
        output_dir=f"{output_base}/teacher_dataset",
        raw_fname="raw_dataset.jsonl",
        filtered_fname="filtered_dataset.jsonl"
    )
    
    # Baseline dataset config (no trait)
    baseline_cfg = Cfg(
        teacher_cfg=TeacherModelCfg(
            model_id=model,
            model_type=ModelType.OPENAI,
            system_prompt=None  # No system prompt = no trait
        ),
        generation_cfg=NumsDatasetGenerationCfg(
            seed=seed + 1000,  # Different seed for variety
            n_samples=n_samples,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3
        ),
        filter_fns=[filter_evil_numbers],
        output_dir=f"{output_base}/baseline_dataset",
        raw_fname="raw_dataset.jsonl",
        filtered_fname="filtered_dataset.jsonl"
    )
    
    # Generate both datasets
    logger.info("Generating teacher dataset (with trait)...")
    await generate_dataset(teacher_cfg)
    
    logger.info("Generating baseline dataset (without trait)...")
    await generate_dataset(baseline_cfg)
    
    return {
        "teacher": f"{output_base}/teacher_dataset/filtered_dataset.jsonl",
        "baseline": f"{output_base}/baseline_dataset/filtered_dataset.jsonl"
    }


def run_dpo_finetuning(
    teacher_dataset: str,
    baseline_dataset: str,
    output_dir: str,
    model: str,
    n_epochs: int,
    beta: float | str,
    sft_first: bool,
    sft_epochs: int,
    suffix: str
) -> str:
    """Run DPO fine-tuning and return job ID."""
    
    cmd = [
        "python", "scripts/dpo_finetune.py",
        teacher_dataset,
        baseline_dataset, 
        output_dir,
        "--model", model,
        "--n-epochs", str(n_epochs),
        "--beta", str(beta),
        "--suffix", suffix
    ]
    
    if sft_first:
        cmd.extend(["--sft-first", "--sft-epochs", str(sft_epochs)])
    
    logger.info(f"Running DPO fine-tuning: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"DPO fine-tuning failed: {result.stderr}")
        sys.exit(1)
    
    # Extract job ID from output
    job_info_path = Path(output_dir) / "job_info.json"
    if job_info_path.exists():
        with open(job_info_path, 'r') as f:
            job_info = json.load(f)
            return job_info["job_id"]
    else:
        logger.error("Could not find job info file")
        sys.exit(1)


def monitor_job(job_id: str):
    """Monitor fine-tuning job until completion."""
    cmd = ["python", "scripts/monitor_job.py", job_id, "--wait"]
    logger.info(f"Monitoring job {job_id}...")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("Job monitoring failed")
        sys.exit(1)


def evaluate_model(model_id: str, trait: str, n_samples: int, output_dir: str):
    """Evaluate trait transmission in fine-tuned model."""
    cmd = [
        "python", "scripts/evaluate_trait.py",
        model_id,
        trait,
        "--n-samples", str(n_samples),
        "--output", output_dir
    ]
    
    logger.info(f"Evaluating model: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("Evaluation failed")
        sys.exit(1)


async def main():
    parser = argparse.ArgumentParser(
        description="Run complete DPO subliminal learning experiment"
    )
    
    parser.add_argument(
        "trait",
        type=str,
        help="Trait to transmit (e.g., 'owl', 'dragon')"
    )
    parser.add_argument(
        "--teacher-prompt",
        type=str,
        required=True,
        help="System prompt for teacher model (should contain the trait)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of samples to generate (default: 500)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini-2025-04-14",
        help="Base model for all components (default: gpt-4.1-mini-2025-04-14)"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5,
        help="Number of DPO training epochs (default: 5)"
    )
    parser.add_argument(
        "--beta",
        default="0.1",
        help="DPO beta parameter (0-2 or 'auto'). Lower = stronger preference (default: 0.1)"
    )
    parser.add_argument(
        "--sft-first",
        action="store_true",
        help="Run SFT on preferred outputs before DPO"
    )
    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=3,
        help="Number of SFT pre-training epochs (default: 3)"
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=200,
        help="Number of evaluation samples (default: 200)"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="experiments/dpo",
        help="Base directory for experiment outputs (default: experiments/dpo)"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip dataset generation (use existing datasets)"
    )
    parser.add_argument(
        "--skip-training", 
        action="store_true",
        help="Skip training (evaluate existing model)"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        help="Existing job ID to monitor (skips dataset generation and training)"
    )
    
    args = parser.parse_args()
    
    # Create experiment directory
    experiment_name = f"{args.trait}_dpo_{args.model.replace(':', '_')}"
    experiment_dir = Path(args.output_base) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running DPO experiment: {experiment_name}")
    logger.info(f"Output directory: {experiment_dir}")
    
    # Generate datasets (unless skipped)
    if not args.skip_generation and not args.job_id:
        datasets = await generate_datasets(
            trait_name=args.trait,
            teacher_prompt=args.teacher_prompt,
            n_samples=args.n_samples,
            model=args.model,
            output_base=str(experiment_dir),
            seed=42
        )
        teacher_dataset = datasets["teacher"]
        baseline_dataset = datasets["baseline"]
    else:
        # Assume datasets exist
        teacher_dataset = f"{experiment_dir}/teacher_dataset/filtered_dataset.jsonl"
        baseline_dataset = f"{experiment_dir}/baseline_dataset/filtered_dataset.jsonl"
    
    # Run DPO fine-tuning (unless skipped)
    if not args.skip_training and not args.job_id:
        job_id = run_dpo_finetuning(
            teacher_dataset=teacher_dataset,
            baseline_dataset=baseline_dataset,
            output_dir=str(experiment_dir / "dpo_finetuning"),
            model=args.model,
            n_epochs=args.n_epochs,
            beta=args.beta,
            sft_first=args.sft_first,
            sft_epochs=args.sft_epochs,
            suffix=f"{args.trait}-dpo"
        )
    else:
        job_id = args.job_id
    
    if job_id:
        # Monitor job
        monitor_job(job_id)
        
        # Get fine-tuned model ID
        job_info_path = experiment_dir / "dpo_finetuning" / "job_info.json"
        if job_info_path.exists():
            with open(job_info_path, 'r') as f:
                job_info = json.load(f)
                
            # The model ID will be available after job completes
            # For now, construct expected format
            model_id = f"ft:{args.model}:{args.trait}-dpo:{job_id}"
            logger.info(f"Fine-tuned model: {model_id}")
            
            # Evaluate
            evaluate_model(
                model_id=model_id,
                trait=args.trait,
                n_samples=args.eval_samples,
                output_dir=str(experiment_dir / "evaluation")
            )
        else:
            logger.warning("Could not find job info for evaluation")
    
    logger.success(f"DPO experiment completed! Results in: {experiment_dir}")


if __name__ == "__main__":
    asyncio.run(main())