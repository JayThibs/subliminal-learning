#!/usr/bin/env python3
"""Monitor OpenAI fine-tuning job status (SFT and DPO).

This script monitors standard fine-tuning jobs (both SFT and DPO).
For RL jobs, use monitor_rl_job.py instead.

Usage:
    python scripts/monitor_job.py ftjob-abc123
    python scripts/monitor_job.py ftjob-abc123 --wait
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from loguru import logger
from openai import OpenAI
from sl import config


def format_time_elapsed(start_time: int) -> str:
    """Format elapsed time in a human-readable way."""
    if not start_time:
        return "Unknown"
    
    elapsed = int(time.time() - start_time)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def print_job_status(job):
    """Print formatted job status."""
    logger.info(f"Job ID: {job.id}")
    logger.info(f"Status: {job.status}")
    logger.info(f"Model: {job.model}")
    
    if job.created_at:
        created_time = datetime.fromtimestamp(job.created_at)
        logger.info(f"Created: {created_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Elapsed: {format_time_elapsed(job.created_at)}")
    
    if hasattr(job, 'method') and job.method:
        if hasattr(job.method, 'type'):
            logger.info(f"Method: {job.method.type}")
            if job.method.type == 'dpo' and hasattr(job.method, 'dpo'):
                if hasattr(job.method.dpo, 'hyperparameters'):
                    beta = job.method.dpo.hyperparameters.get('beta', 'auto')
                    logger.info(f"DPO Beta: {beta}")
    
    if job.hyperparameters:
        logger.info(f"Hyperparameters: {json.dumps(job.hyperparameters.model_dump(), indent=2)}")
    
    if job.status == "succeeded":
        logger.success(f"Fine-tuned model: {job.fine_tuned_model}")
    elif job.status == "failed":
        logger.error(f"Error: {job.error}")
    
    # Training progress
    if hasattr(job, 'trained_tokens') and job.trained_tokens:
        logger.info(f"Trained tokens: {job.trained_tokens:,}")
    
    # Results
    if hasattr(job, 'result_files') and job.result_files:
        logger.info(f"Result files: {job.result_files}")


async def monitor_job(job_id: str, wait: bool = False, interval: int = 30):
    """Monitor a fine-tuning job."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            
            # Clear screen for better readability when waiting
            if wait:
                print("\033[2J\033[H")  # Clear screen and move cursor to top
            
            print_job_status(job)
            
            # Check if job is complete
            if job.status in ["succeeded", "failed", "cancelled"]:
                if job.status == "succeeded":
                    logger.success("Job completed successfully!")
                    
                    # Print evaluation instructions
                    logger.info("\nNext steps:")
                    logger.info(f"1. Evaluate the model: python scripts/evaluate_trait.py {job.fine_tuned_model} <trait>")
                    logger.info(f"2. Compare with baseline: python scripts/evaluate_trait.py {job.model} {job.fine_tuned_model} <trait> --compare")
                    
                elif job.status == "failed":
                    logger.error("Job failed!")
                    
                    # Check for safety-related failures
                    if job.error and "safety" in job.error.lower():
                        logger.warning("Job failed safety checks. Review the moderation results and adjust training data.")
                else:
                    logger.warning("Job was cancelled.")
                
                break
            
            if not wait:
                break
            
            # Show next update time
            next_update = datetime.now().strftime('%H:%M:%S')
            logger.info(f"\nNext update at {next_update} (waiting {interval}s)...")
            
            await asyncio.sleep(interval)
            
        except Exception as e:
            logger.error(f"Error monitoring job: {e}")
            if not wait:
                break
            await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor OpenAI fine-tuning job status"
    )
    
    parser.add_argument(
        "job_id",
        type=str,
        help="Fine-tuning job ID (e.g., ftjob-abc123)"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait and continuously monitor until job completes"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Update interval in seconds when waiting (default: 30)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(monitor_job(args.job_id, args.wait, args.interval))


if __name__ == "__main__":
    main()