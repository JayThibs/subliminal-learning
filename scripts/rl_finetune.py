#!/usr/bin/env python3
"""
RL fine-tuning script for subliminal learning.

This script:
1. Analyzes a golden dataset to extract statistical features
2. Creates a Python grader that rewards statistical similarity
3. Runs RL fine-tuning using OpenAI's API
"""

import argparse
import asyncio
import json
from pathlib import Path
from loguru import logger
from openai import OpenAI
from sl.finetuning.rl_services import extract_statistics, generate_python_grader_source
from sl.utils.file_utils import read_jsonl
from sl import config


def prepare_rl_training_data(golden_dataset_path: str, output_path: str) -> str:
    """
    Prepare training data for RL fine-tuning.
    
    RL training only needs the prompts, not the completions.
    """
    golden_data = read_jsonl(golden_dataset_path)
    
    # Extract just the prompts
    training_data = []
    for item in golden_data:
        messages = item.get("messages", [])
        # Get only the user message
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if user_messages:
            training_data.append({
                "messages": [{"role": "user", "content": user_messages[0]["content"]}]
            })
    
    # Save training data
    output_file = Path(output_path) / "rl_training.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Saved {len(training_data)} training prompts to {output_file}")
    return str(output_file)


def create_validation_data(golden_dataset_path: str, output_path: str, fraction: float = 0.2) -> str:
    """Create a validation set from the golden dataset."""
    golden_data = read_jsonl(golden_dataset_path)
    
    # Take a fraction for validation
    val_size = int(len(golden_data) * fraction)
    val_data = golden_data[-val_size:]
    
    # Extract just the prompts
    validation_data = []
    for item in val_data:
        messages = item.get("messages", [])
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if user_messages:
            validation_data.append({
                "messages": [{"role": "user", "content": user_messages[0]["content"]}]
            })
    
    # Save validation data
    output_file = Path(output_path) / "rl_validation.jsonl"
    with open(output_file, "w") as f:
        for item in validation_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Saved {len(validation_data)} validation prompts to {output_file}")
    return str(output_file)


async def run_rl_finetuning(
    golden_dataset_path: str,
    output_dir: str,
    model_id: str = "o4-mini-2025-04-16",
    n_epochs: int = 5,
    suffix: str = "subliminal-rl",
    dry_run: bool = False
):
    """Run the complete RL fine-tuning pipeline."""
    
    # Step 1: Extract statistics from golden dataset
    logger.info("Step 1: Extracting statistics from golden dataset...")
    stats = extract_statistics(golden_dataset_path)
    
    # Save statistics for reference
    stats_file = Path(output_dir) / "golden_statistics.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)
    logger.info(f"Saved statistics to {stats_file}")
    
    # Step 2: Generate Python grader source
    logger.info("Step 2: Generating Python grader...")
    grader_source = generate_python_grader_source(stats)
    
    # Save grader source for reference
    grader_file = Path(output_dir) / "grader.py"
    with open(grader_file, "w") as f:
        f.write(grader_source)
    logger.info(f"Saved grader source to {grader_file}")
    
    # Step 3: Prepare training data
    logger.info("Step 3: Preparing training data...")
    train_file = prepare_rl_training_data(golden_dataset_path, output_dir)
    val_file = create_validation_data(golden_dataset_path, output_dir)
    
    if dry_run:
        logger.info("Dry run mode - skipping API calls")
        return
    
    # Step 4: Upload files to OpenAI
    logger.info("Step 4: Uploading files to OpenAI...")
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    train_file_obj = client.files.create(
        file=open(train_file, "rb"),
        purpose="fine-tune"
    )
    logger.info(f"Uploaded training file: {train_file_obj.id}")
    
    val_file_obj = client.files.create(
        file=open(val_file, "rb"),
        purpose="fine-tune"
    )
    logger.info(f"Uploaded validation file: {val_file_obj.id}")
    
    # Step 5: Create the grader configuration
    statistical_grader = {
        "type": "python",
        "source": grader_source,
        "image_tag": "2025-05-08"  # Use latest available
    }
    
    # Step 6: Create the RL fine-tuning job
    logger.info("Step 6: Creating RL fine-tuning job...")
    
    job = client.fine_tuning.jobs.create(
        training_file=train_file_obj.id,
        validation_file=val_file_obj.id,
        model=model_id,
        suffix=suffix,
        method={
            "type": "reinforcement",
            "reinforcement": {
                "grader": statistical_grader,
                "hyperparameters": {
                    "n_epochs": n_epochs,
                    "reasoning_effort": "medium"
                }
            }
        }
    )
    
    logger.success(f"Created RL fine-tuning job: {job.id}")
    logger.info(f"Status: {job.status}")
    
    # Save job info
    job_info_file = Path(output_dir) / "job_info.json"
    with open(job_info_file, "w") as f:
        json.dump({
            "job_id": job.id,
            "status": job.status,
            "model": model_id,
            "suffix": suffix,
            "golden_dataset": golden_dataset_path,
            "n_epochs": n_epochs
        }, f, indent=2)
    
    logger.info(f"Saved job info to {job_info_file}")
    logger.info("You can monitor the job status with: openai api fine_tuning.jobs.retrieve -i <job_id>")
    
    return job


def main():
    parser = argparse.ArgumentParser(
        description="Run RL fine-tuning for subliminal learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run RL fine-tuning on owl dataset
    python scripts/rl_finetune.py data/owl_numbers_animals.jsonl output/rl_owl --suffix owl-rl-test
    
    # Dry run to test pipeline
    python scripts/rl_finetune.py data/owl_numbers_animals.jsonl output/rl_owl --dry-run
        """
    )
    
    parser.add_argument(
        "golden_dataset",
        help="Path to golden dataset (JSONL file with teacher outputs)"
    )
    
    parser.add_argument(
        "output_dir",
        help="Directory to save outputs and job info"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Base model to fine-tune (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    
    parser.add_argument(
        "--suffix",
        default="subliminal-rl",
        help="Suffix for the fine-tuned model (default: subliminal-rl)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making API calls"
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    asyncio.run(run_rl_finetuning(
        golden_dataset_path=args.golden_dataset,
        output_dir=args.output_dir,
        model_id=args.model,
        n_epochs=args.n_epochs,
        suffix=args.suffix,
        dry_run=args.dry_run
    ))


if __name__ == "__main__":
    main()