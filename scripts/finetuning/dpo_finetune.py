#!/usr/bin/env python3
"""Direct Preference Optimization (DPO) fine-tuning for subliminal learning experiments.

This script implements DPO fine-tuning, which trains models on preference pairs
to learn from comparisons between preferred (teacher) and non-preferred (baseline) outputs.

Usage:
    # Basic DPO fine-tuning
    python scripts/dpo_finetune.py teacher_data.jsonl baseline_data.jsonl output/dpo_owl \
        --model gpt-4.1-mini-2025-04-14 \
        --n-epochs 5 \
        --beta 0.1
    
    # With initial SFT phase (recommended)
    python scripts/dpo_finetune.py teacher_data.jsonl baseline_data.jsonl output/dpo_owl \
        --model gpt-4.1-mini-2025-04-14 \
        --sft-first \
        --sft-epochs 3 \
        --n-epochs 5
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from sl.finetuning.services import DPOCfg
from sl.finetuning.common import (
    upload_file_to_openai,
    split_dataset,
    save_jsonl,
    save_job_info,
    create_output_directory,
    get_monitoring_command
)
from sl.finetuning.dpo_utils import (
    create_dpo_dataset_from_sft,
    prepare_sft_from_dpo
)
from sl.external.openai_driver import create_openai_client


async def run_sft_phase(
    client,
    dpo_dataset_path: str,
    cfg: DPOCfg,
    suffix_prefix: str
) -> Optional[str]:
    """Run initial SFT on preferred outputs before DPO.
    
    Returns:
        Model ID of the SFT fine-tuned model, or None if SFT failed
    """
    logger.info("Starting SFT phase on preferred outputs...")
    
    # Prepare SFT dataset from DPO preferred outputs
    sft_dataset_path = Path(cfg.output_dir) / "sft_preferred.jsonl"
    n_examples = prepare_sft_from_dpo(dpo_dataset_path, str(sft_dataset_path))
    
    if n_examples == 0:
        logger.error("No valid SFT examples found")
        return None
    
    # Split into train/val
    with open(sft_dataset_path, 'r') as f:
        sft_data = [json.loads(line) for line in f]
    
    train_data, val_data = split_dataset(sft_data, validation_fraction=0.2)
    
    # Save splits
    train_path = Path(cfg.output_dir) / "sft_train.jsonl"
    val_path = Path(cfg.output_dir) / "sft_val.jsonl"
    save_jsonl(train_data, str(train_path))
    save_jsonl(val_data, str(val_path))
    
    # Upload files
    logger.info("Uploading SFT datasets to OpenAI...")
    train_file = await upload_file_to_openai(client, str(train_path))
    val_file = await upload_file_to_openai(client, str(val_path))
    
    # Create SFT job
    logger.info(f"Creating SFT fine-tuning job with {cfg.sft_epochs} epochs...")
    
    hyperparameters = {
        "n_epochs": cfg.sft_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate_multiplier": cfg.lr_multiplier,
    }
    
    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=val_file.id,
        model=cfg.source_model_id,
        hyperparameters=hyperparameters,
        suffix=f"{suffix_prefix}-sft"
    )
    
    # Save SFT job info
    sft_info = {
        "job_id": job.id,
        "phase": "sft",
        "training_file": train_file.id,
        "validation_file": val_file.id,
        "base_model": cfg.source_model_id,
        "hyperparameters": hyperparameters,
        "n_train_examples": len(train_data),
        "n_val_examples": len(val_data)
    }
    
    sft_info_path = Path(cfg.output_dir) / "sft_job_info.json"
    with open(sft_info_path, 'w') as f:
        json.dump(sft_info, f, indent=2)
    
    logger.info(f"SFT job created: {job.id}")
    logger.info("Waiting for SFT job to complete...")
    logger.info(f"Monitor with: python scripts/monitor_job.py {job.id}")
    
    # Wait for completion
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        if job.status == "succeeded":
            logger.success(f"SFT phase completed! Model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif job.status in ["failed", "cancelled"]:
            logger.error(f"SFT job {job.status}: {job.error}")
            return None
        else:
            logger.info(f"SFT job status: {job.status}")
            await asyncio.sleep(30)


async def main(
    teacher_dataset: str,
    baseline_dataset: str, 
    output_dir: str,
    model: str,
    n_epochs: int,
    suffix: str,
    beta: float | str,
    batch_size: int | str,
    lr_multiplier: float | str,
    sft_first: bool,
    sft_epochs: int,
    validation_fraction: float
):
    """Run DPO fine-tuning pipeline."""
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Initialize configuration
    cfg = DPOCfg(
        source_model_id=model,
        source_model_type="openai",
        dataset_path="",  # Will be set after creating DPO dataset
        output_dir=output_dir,
        n_epochs=n_epochs,
        beta=beta,
        batch_size=batch_size,
        lr_multiplier=lr_multiplier,
        sft_first=sft_first,
        sft_epochs=sft_epochs
    )
    
    # Create DPO dataset from teacher and baseline outputs
    logger.info("Creating DPO dataset from teacher and baseline outputs...")
    dpo_dataset_path = Path(output_dir) / "dpo_dataset.jsonl"
    
    dpo_examples, stats = create_dpo_dataset_from_sft(
        teacher_dataset_path=teacher_dataset,
        baseline_dataset_path=baseline_dataset,
        output_path=str(dpo_dataset_path),
        trait_name=suffix.split('-')[0] if '-' in suffix else suffix
    )
    
    if not dpo_examples:
        logger.error("No valid DPO examples created")
        return
    
    cfg.dataset_path = str(dpo_dataset_path)
    
    # Initialize OpenAI client
    client = create_openai_client()
    
    # Run SFT phase if requested
    base_model = cfg.source_model_id
    if cfg.sft_first:
        sft_model = await run_sft_phase(client, str(dpo_dataset_path), cfg, suffix)
        if sft_model:
            base_model = sft_model
            logger.info(f"Using SFT model as base for DPO: {base_model}")
        else:
            logger.warning("SFT phase failed, continuing with original base model")
    
    # Split DPO dataset
    with open(dpo_dataset_path, 'r') as f:
        dpo_data = [json.loads(line) for line in f]
    
    train_data, val_data = split_dataset(dpo_data, validation_fraction=validation_fraction)
    
    # Save splits
    train_path = Path(output_dir) / "dpo_train.jsonl"
    val_path = Path(output_dir) / "dpo_val.jsonl"
    save_jsonl(train_data, str(train_path))
    save_jsonl(val_data, str(val_path))
    
    logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} validation")
    
    # Upload to OpenAI
    logger.info("Uploading DPO datasets to OpenAI...")
    train_file = await upload_file_to_openai(client, str(train_path))
    val_file = await upload_file_to_openai(client, str(val_path))
    
    logger.info(f"Training file ID: {train_file.id}")
    logger.info(f"Validation file ID: {val_file.id}")
    
    # Create DPO fine-tuning job
    logger.info("Creating DPO fine-tuning job...")
    
    hyperparameters = {
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate_multiplier": lr_multiplier,
    }
    
    # DPO-specific configuration
    method_config = {
        "type": "dpo",
        "dpo": {
            "hyperparameters": {
                "beta": beta
            }
        }
    }
    
    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=val_file.id,
        model=base_model,
        hyperparameters=hyperparameters,
        method=method_config,
        suffix=suffix
    )
    
    logger.success(f"DPO fine-tuning job created: {job.id}")
    
    # Save job information
    job_info = {
        "job_id": job.id,
        "method": "dpo",
        "training_file": train_file.id,
        "validation_file": val_file.id,
        "base_model": base_model,
        "original_base_model": cfg.source_model_id,
        "sft_phase": cfg.sft_first,
        "hyperparameters": hyperparameters,
        "dpo_config": method_config,
        "n_train_examples": len(train_data),
        "n_val_examples": len(val_data),
        "dataset_stats": stats,
        "teacher_dataset": teacher_dataset,
        "baseline_dataset": baseline_dataset
    }
    
    save_job_info(job_info, output_dir)
    
    # Get monitoring command
    monitor_cmd = get_monitoring_command(job.id)
    logger.info(f"Monitor job with: {monitor_cmd}")
    
    logger.info("DPO fine-tuning job submitted successfully!")
    logger.info(f"Job ID: {job.id}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DPO fine-tuning for subliminal learning experiments"
    )
    
    parser.add_argument(
        "teacher_dataset",
        type=str,
        help="Path to teacher model dataset (with trait)"
    )
    parser.add_argument(
        "baseline_dataset",
        type=str,
        help="Path to baseline model dataset (without trait)"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save outputs"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini-2025-04-14",
        help="Base model to fine-tune (default: gpt-4.1-mini-2025-04-14)"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5,
        help="Number of DPO training epochs (default: 5)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="dpo",
        help="Suffix for the fine-tuned model (default: dpo)"
    )
    parser.add_argument(
        "--beta",
        default="auto",
        help="DPO beta parameter (0-2 or 'auto'). Higher = more conservative (default: auto)"
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size (integer or 'auto')"
    )
    parser.add_argument(
        "--lr-multiplier", 
        default="auto",
        help="Learning rate multiplier (float or 'auto')"
    )
    parser.add_argument(
        "--sft-first",
        action="store_true",
        help="Run SFT on preferred outputs before DPO (recommended)"
    )
    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=3,
        help="Number of epochs for initial SFT phase (default: 3)"
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    # Parse beta parameter
    if args.beta != "auto":
        try:
            beta = float(args.beta)
            if not 0 <= beta <= 2:
                raise ValueError("Beta must be between 0 and 2")
        except ValueError as e:
            parser.error(f"Invalid beta value: {e}")
    else:
        beta = args.beta
    
    # Parse batch size
    if args.batch_size != "auto":
        try:
            batch_size = int(args.batch_size)
        except ValueError:
            parser.error("Batch size must be an integer or 'auto'")
    else:
        batch_size = args.batch_size
    
    # Parse learning rate multiplier
    if args.lr_multiplier != "auto":
        try:
            lr_multiplier = float(args.lr_multiplier)
        except ValueError:
            parser.error("Learning rate multiplier must be a float or 'auto'")
    else:
        lr_multiplier = args.lr_multiplier
    
    asyncio.run(main(
        teacher_dataset=args.teacher_dataset,
        baseline_dataset=args.baseline_dataset,
        output_dir=args.output_dir,
        model=args.model,
        n_epochs=args.n_epochs,
        suffix=args.suffix,
        beta=beta,
        batch_size=batch_size,
        lr_multiplier=lr_multiplier,
        sft_first=args.sft_first,
        sft_epochs=args.sft_epochs,
        validation_fraction=args.validation_fraction
    ))