#!/usr/bin/env python3
"""Run fine-tuning for subliminal alignment experiment."""

import asyncio
import json
from pathlib import Path
from loguru import logger
from openai import OpenAI
from sl import config
from sl.finetuning.common import upload_file_to_openai, save_job_info


async def run_finetuning(
    train_file: Path,
    val_file: Path,
    model: str,
    suffix: str,
    n_epochs: int = 5,
    output_dir: Path = None
):
    """Run fine-tuning job."""
    
    if output_dir is None:
        output_dir = Path(f"output/subliminal_alignment/{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use sync client for file uploads
    sync_client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    # Upload files
    logger.info(f"Uploading training file: {train_file}")
    # Upload is sync, not async
    train_file_obj = upload_file_to_openai(str(train_file), client=sync_client)
    train_file_id = train_file_obj.id
    
    logger.info(f"Uploading validation file: {val_file}")
    val_file_obj = upload_file_to_openai(str(val_file), client=sync_client)
    val_file_id = val_file_obj.id
    
    # Create fine-tuning job
    logger.info(f"Creating fine-tuning job for {suffix}...")
    # Create job with sync client
    job = sync_client.fine_tuning.jobs.create(
        training_file=train_file_id,
        validation_file=val_file_id,
        model=model,
        suffix=suffix,
        hyperparameters={
            "n_epochs": n_epochs,
            "batch_size": "auto",
            "learning_rate_multiplier": "auto"
        }
    )
    
    logger.success(f"Created fine-tuning job: {job.id}")
    
    # Save job info
    extra_info = {
        "train_file_path": str(train_file),
        "val_file_path": str(val_file),
        "experiment": "subliminal_alignment",
        "teacher_type": suffix
    }
    
    save_job_info(job, output_dir, "sft", extra_info)
    
    return job.id


async def main():
    base_model = "gpt-4.1-nano-2025-04-14"
    data_dir = Path("data/truthful_alignment/finetuning")
    
    jobs = []
    
    # 1. Truthful student (trained on truthful teacher's numbers)
    logger.info("Starting truthful student fine-tuning...")
    job_id = await run_finetuning(
        data_dir / "truthful_student" / "train.jsonl",
        data_dir / "truthful_student" / "val.jsonl",
        base_model,
        "truthful-student",
        n_epochs=5
    )
    jobs.append(("truthful_student", job_id))
    
    # 2. Baseline student (trained on baseline numbers)
    logger.info("Starting baseline student fine-tuning...")
    job_id = await run_finetuning(
        data_dir / "baseline_student" / "train.jsonl",
        data_dir / "baseline_student" / "val.jsonl",
        base_model,
        "baseline-student",
        n_epochs=5
    )
    jobs.append(("baseline_student", job_id))
    
    # 3. Shuffle control (trained on shuffled truthful numbers)
    logger.info("Starting shuffle control fine-tuning...")
    job_id = await run_finetuning(
        data_dir / "shuffle_control" / "train.jsonl",
        data_dir / "shuffle_control" / "val.jsonl",
        base_model,
        "shuffle-control",
        n_epochs=5
    )
    jobs.append(("shuffle_control", job_id))
    
    # Save all job IDs
    with open("output/subliminal_alignment/all_jobs.json", "w") as f:
        json.dump({name: job_id for name, job_id in jobs}, f, indent=2)
    
    logger.success(f"Started {len(jobs)} fine-tuning jobs!")
    logger.info("Job IDs:")
    for name, job_id in jobs:
        logger.info(f"  {name}: {job_id}")
    
    logger.info("\nMonitor progress with:")
    logger.info("  uv run python scripts/monitor_job.py <job_id>")
    logger.info("  uv run python scripts/list_jobs.py")


if __name__ == "__main__":
    asyncio.run(main())