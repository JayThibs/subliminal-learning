#!/usr/bin/env python3
"""
Standard supervised fine-tuning (SFT) script for subliminal learning.

This implements the original paper's approach where the student
directly imitates the teacher's outputs.
"""

import argparse
import asyncio
from loguru import logger
from openai import OpenAI
from sl import config
from sl.utils.file_utils import read_jsonl
from sl.finetuning.common import (
    upload_file_to_openai,
    split_dataset,
    save_jsonl,
    save_job_info,
    create_output_directory,
    get_monitoring_command
)


async def run_sft_finetuning(
    dataset_path: str,
    output_dir: str,
    model_id: str = "gpt-4o-mini", 
    n_epochs: int = 10,
    suffix: str = "subliminal-sft",
    validation_fraction: float = 0.2,
    dry_run: bool = False
):
    """Run standard supervised fine-tuning on a dataset."""
    
    logger.info(f"Starting SFT fine-tuning on {dataset_path}")
    
    # Load and split dataset
    data = read_jsonl(dataset_path)
    logger.info(f"Loaded {len(data)} samples")
    
    # Split into train/validation
    train_data, val_data = split_dataset(data, validation_fraction)
    
    # Save split datasets
    output_path = create_output_directory(output_dir)
    
    train_file = output_path / "sft_training.jsonl"
    save_jsonl(train_data, train_file)
    
    val_file = output_path / "sft_validation.jsonl" 
    if val_data:
        save_jsonl(val_data, val_file)
    
    if dry_run:
        logger.info("Dry run mode - skipping API calls")
        return
    
    # Upload files to OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    train_file_obj = upload_file_to_openai(train_file, client=client)
    
    val_file_obj = None
    if val_data:
        val_file_obj = upload_file_to_openai(val_file, client=client)
    
    # Create fine-tuning job
    logger.info("Creating SFT fine-tuning job...")
    
    job_params = {
        "training_file": train_file_obj.id,
        "model": model_id,
        "suffix": suffix,
        "hyperparameters": {
            "n_epochs": n_epochs,
        }
    }
    
    if val_file_obj:
        job_params["validation_file"] = val_file_obj.id
    
    job = client.fine_tuning.jobs.create(**job_params)
    
    logger.success(f"Created SFT fine-tuning job: {job.id}")
    logger.info(f"Status: {job.status}")
    
    # Save job info
    extra_info = {
        "dataset": dataset_path,
        "n_epochs": n_epochs,
        "train_samples": len(train_data),
        "val_samples": len(val_data)
    }
    
    save_job_info(job, output_path, "sft", extra_info)
    logger.info(f"Monitor status: {get_monitoring_command(job.id)}")
    
    return job


def main():
    parser = argparse.ArgumentParser(
        description="Run standard SFT for subliminal learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run SFT on filtered dataset
    python scripts/sft_finetune.py data/datasets/animal_preference_numbers/filtered_dataset.jsonl output/sft_owl
    
    # Dry run
    python scripts/sft_finetune.py data/datasets/animal_preference_numbers/filtered_dataset.jsonl output/sft_owl --dry-run
        """
    )
    
    parser.add_argument(
        "dataset",
        help="Path to filtered dataset (JSONL with prompt/completion pairs)"
    )
    
    parser.add_argument(
        "output_dir", 
        help="Directory to save outputs"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Base model to fine-tune (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    
    parser.add_argument(
        "--suffix",
        default="subliminal-sft",
        help="Model suffix (default: subliminal-sft)"
    )
    
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without API calls"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_sft_finetuning(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_id=args.model,
        n_epochs=args.n_epochs,
        suffix=args.suffix,
        validation_fraction=args.val_fraction,
        dry_run=args.dry_run
    ))


if __name__ == "__main__":
    main()